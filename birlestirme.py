import cv2
import numpy as np

from errors import PanoramaError

# A panorama of two overlapping photos stays within a few times the input
# area. A canvas larger than this multiple of the total input pixel count
# means the homography is degenerate, so refuse it before allocating it.
MAX_TUVAL_KATSAYISI = 8

# Interpolating a warp mixes the black canvas background into the pixels just
# inside the image border, so the outermost ring of a warped image is a dark
# fringe rather than real content. Warped validity masks are eroded by this
# radius to drop it.
MASKE_ASINDIRMA = 2

# Bounds for the per-channel exposure gain. A real exposure difference between
# two shots of the same scene is small; anything outside this range says the
# overlap statistics are not comparable (moving subject, clipped highlights),
# and scaling by it would do more harm than the mismatch it corrects.
POZLAMA_ALT_SINIR = 0.8
POZLAMA_UST_SINIR = 1.25

# Feather band around the seam. The seam runs along the middle of the overlap
# (equidistant from both image borders) and the two sources are mixed only
# within this half-width of it, proportional to the overlap width and clamped
# to a pixel range. Mixing the whole overlap instead would blend anything that
# moved between the two shots into a semi-transparent ghost.
DIKIS_BANT_PAYI = 15
DIKIS_BANT_EN_AZ = 15
DIKIS_BANT_EN_COK = 60

# Auto-crop: an edge row/column is trimmed only when it is emptier than this
# ratio AND emptier than the box it borders, and never past this fraction of
# the original size. See _gecerli_alani_kirp.
KIRPMA_ESIGI = 0.8
KIRPMA_EN_AZ_ORAN = 0.25


def _homografiyi_dogrula(H):
    """Reject homographies that cannot produce a usable canvas."""
    if H is None:
        raise PanoramaError("Homografi yok, birlestirme yapilamadi.")

    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        raise PanoramaError("Homografi dejenere; goruntuler yeterince ortusmuyor olabilir.")

    # A near-singular matrix collapses the image onto a line or blows it up.
    if abs(np.linalg.det(H)) < 1e-8:
        raise PanoramaError("Homografi dejenere; goruntuler yeterince ortusmuyor olabilir.")
    return H


def _tuval_ve_cevirme(H, sol_yukseklik, sol_genislik, sag_yukseklik, sag_genislik, pad=2):
    """Compute canvas size and translation from warped left and fixed right image."""
    _homografiyi_dogrula(H)

    sol_kose = np.float32(
        [[0, 0], [sol_genislik, 0], [sol_genislik, sol_yukseklik], [0, sol_yukseklik]]
    ).reshape(-1, 1, 2)
    sol_donusmus = cv2.perspectiveTransform(sol_kose, H).reshape(-1, 2)

    sag_kose = np.float32(
        [[0, 0], [sag_genislik, 0], [sag_genislik, sag_yukseklik], [0, sag_yukseklik]]
    )
    tum_noktalar = np.vstack([sol_donusmus, sag_kose])

    # Points behind the camera plane come back as inf/nan; never feed those
    # into the int() conversions below.
    if not np.all(np.isfinite(tum_noktalar)):
        raise PanoramaError("Homografi dejenere; goruntuler yeterince ortusmuyor olabilir.")

    xmin, ymin = tum_noktalar.min(axis=0)
    xmax, ymax = tum_noktalar.max(axis=0)

    # Check the size in float first: a degenerate H can ask for a canvas of
    # billions of pixels, and even computing it as int is pointless then.
    girdi_pikselleri = sol_yukseklik * sol_genislik + sag_yukseklik * sag_genislik
    istenen_alan = float(xmax - xmin + 2 * pad) * float(ymax - ymin + 2 * pad)
    if istenen_alan > MAX_TUVAL_KATSAYISI * girdi_pikselleri:
        raise PanoramaError("Homografi dejenere; goruntuler yeterince ortusmuyor olabilir.")

    tx = int(np.floor(-xmin)) + pad
    ty = int(np.floor(-ymin)) + pad
    tuval_genislik = int(np.ceil(xmax - xmin)) + 2 * pad
    tuval_yukseklik = int(np.ceil(ymax - ymin)) + 2 * pad

    T = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    return tuval_genislik, tuval_yukseklik, T


def _gecerli_maske(goruntu):
    """Separate valid image pixels from black background.

    Only a fallback for callers that have no warped mask at hand: a dark but
    real scene pixel is indistinguishable from background this way, and the
    interpolation fringe around a warp counts as valid. Prefer the masks
    produced by _warp_maskesi / _yerlestirme_maskesi.
    """
    return np.any(goruntu > 0, axis=2).astype(np.uint8) * 255


def _maske_asindir(maske, yaricap=MASKE_ASINDIRMA):
    """Shave `yaricap` pixels off a 0/255 mask to drop the interpolation fringe.

    cv2.erode leaves the canvas border untouched, which is what we want here:
    where the warped image runs past the edge of the canvas, the pixels at that
    edge are interior image content and carry no fringe. An image so small that
    the erosion would erase it keeps its mask instead.
    """
    if yaricap <= 0:
        return maske
    cekirdek = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * yaricap + 1, 2 * yaricap + 1))
    asinmis = cv2.erode(maske, cekirdek)
    if cv2.countNonZero(asinmis) == 0:
        return maske
    return asinmis


def _warp_maskesi(yukseklik, genislik, M, tuval_genislik, tuval_yukseklik):
    """Validity mask of an image warped by M, as an eroded 0/255 canvas mask.

    The mask is warped with INTER_NEAREST so it stays binary, then eroded: the
    image itself is resampled with a smoothing kernel that pulls the black
    background into the border pixels, and those must not count as content.
    """
    dolu = np.full((yukseklik, genislik), 255, dtype=np.uint8)
    maske = cv2.warpPerspective(dolu, M, (tuval_genislik, tuval_yukseklik), flags=cv2.INTER_NEAREST)
    return _maske_asindir(maske)


def _yerlestirme_maskesi(yukseklik, genislik, tx, ty, tuval_genislik, tuval_yukseklik):
    """Validity mask of an image copied into the canvas at an integer offset.

    A slice copy resamples nothing, so this rectangle is exact and needs no
    erosion -- unlike the warped mask above.
    """
    dolu = np.full((yukseklik, genislik), 255, dtype=np.uint8)
    return _cevirerek_yerlestir(dolu, tx, ty, tuval_genislik, tuval_yukseklik)


def _kutu(maske):
    """Bounding box (ust, alt, sol, sag) of the non-zero pixels, end-exclusive."""
    satirlar = np.flatnonzero(maske.any(axis=1))
    if satirlar.size == 0:
        return None
    sutunlar = np.flatnonzero(maske.any(axis=0))
    return int(satirlar[0]), int(satirlar[-1]) + 1, int(sutunlar[0]), int(sutunlar[-1]) + 1


def _pozlama_esitle(warp_sol, warp_sag, maske_sol, maske_sag):
    """Scale the warped left image so its overlap exposure matches the right one.

    Two shots of the same scene rarely share an exposure; the gain is a single
    per-channel factor measured on the overlap, applied to the whole left image
    through a lookup table so the corrected part never disagrees with the rest.
    """
    ortusme = cv2.bitwise_and(maske_sol, maske_sag)
    if cv2.countNonZero(ortusme) == 0:
        return warp_sol

    sol_ortalama = cv2.mean(warp_sol, mask=ortusme)[:3]
    sag_ortalama = cv2.mean(warp_sag, mask=ortusme)[:3]

    kazanclar = []
    for sol_kanal, sag_kanal in zip(sol_ortalama, sag_ortalama, strict=True):
        # A near-black overlap carries no exposure information; dividing by it
        # would produce a huge gain from rounding noise alone.
        if sol_kanal < 1.0:
            kazanclar.append(1.0)
            continue
        kazanclar.append(
            float(np.clip(sag_kanal / sol_kanal, POZLAMA_ALT_SINIR, POZLAMA_UST_SINIR))
        )

    if all(abs(kazanc - 1.0) < 1e-3 for kazanc in kazanclar):
        return warp_sol

    olcekli = np.arange(256, dtype=np.float32)[None, :, None] * np.float32(kazanclar)
    lut = np.clip(np.rint(olcekli), 0, 255).astype(np.uint8)
    return cv2.LUT(warp_sol, lut)


def _dikis_yari_banti(ortusme_genisligi):
    """Half-width in pixels of the mixing band around the seam."""
    return int(np.clip(ortusme_genisligi // DIKIS_BANT_PAYI, DIKIS_BANT_EN_AZ, DIKIS_BANT_EN_COK))


def _mesafe_agirligi(maske_sol, maske_sag, kutu):
    """Feather weights for the left image inside the overlap box.

    d_sol and d_sag are each pixel's distances to the border of the left and
    right image. Where they are equal lies the seam: the line that stays as
    far as possible from both image borders, so neither outline ever shows up
    as a hard edge and the direction of the ramp follows the geometry (a left
    image that actually landed on the right is handled by the same formula).
    Moving away from the seam, d_sol - d_sag changes by about two per pixel, so
    the weight reaches 0 or 1 exactly `yari_bant` pixels out; outside the band
    each side is a single source, which keeps anything that moved between the
    two shots from being blended into a ghost.

    The distance transforms are computed on the whole canvas -- cropping first
    would make the box edges look like image borders and ramp the weights in
    the wrong place -- but only the box is kept, one at a time, so at most one
    full-canvas float array exists at a time.
    """
    ust, alt, sol, sag = kutu
    d_sol = cv2.distanceTransform(maske_sol, cv2.DIST_L2, 3)[ust:alt, sol:sag].copy()
    d_sag = cv2.distanceTransform(maske_sag, cv2.DIST_L2, 3)[ust:alt, sol:sag].copy()

    yari_bant = _dikis_yari_banti(sag - sol)
    d_sol -= d_sag
    d_sol /= 4.0 * yari_bant
    d_sol += 0.5
    return np.clip(d_sol, 0.0, 1.0, out=d_sol)


def _tuy_birlestir(warp_sol, warp_sag, maske_sol=None, maske_sag=None):
    """
    Feather-blend two warped BGR images along the seam of their overlap.

    The seam is the line equidistant from both image borders and the sources
    are mixed only in a narrow band around it (see _mesafe_agirligi), which
    makes the transition independent of image content and of which side the
    left image actually landed on, keeps the warped outline from showing as a
    hard edge, and leaves moving subjects unblended outside the band.

    Float work is confined to the bounding box of the overlap; everything
    outside it is a uint8 copy.
    """
    if maske_sol is None:
        maske_sol = _gecerli_maske(warp_sol)
    if maske_sag is None:
        maske_sag = _gecerli_maske(warp_sag)

    # Exclusive regions first, as a plain uint8 masked copy: the right image
    # overwrites the overlap, which the feathered mix then replaces.
    sonuc = np.zeros_like(warp_sol)
    sonuc = cv2.copyTo(warp_sol, maske_sol, sonuc)
    sonuc = cv2.copyTo(warp_sag, maske_sag, sonuc)

    ortusme = cv2.bitwise_and(maske_sol, maske_sag)
    kutu = _kutu(ortusme)
    if kutu is None:
        return sonuc

    ust, alt, sol, sag = kutu
    agirlik = _mesafe_agirligi(maske_sol, maske_sag, kutu)[..., None]

    karisim = warp_sol[ust:alt, sol:sag].astype(np.float32)
    karisim *= agirlik
    sag_pay = warp_sag[ust:alt, sol:sag].astype(np.float32)
    sag_pay *= 1.0 - agirlik
    karisim += sag_pay
    del sag_pay

    np.copyto(
        sonuc[ust:alt, sol:sag],
        np.clip(karisim, 0, 255).astype(np.uint8),
        where=(ortusme[ust:alt, sol:sag] > 0)[..., None],
    )
    return sonuc


def _bolge_toplami(toplam, ust, alt, sol, sag):
    """Valid pixel count inside the inclusive box, read off the integral image."""
    return int(
        toplam[alt + 1, sag + 1] - toplam[ust, sag + 1] - toplam[alt + 1, sol] + toplam[ust, sol]
    )


def _satir_orani(toplam, satir, sol, sag):
    """Occupancy ratio of one row over the column range [sol, sag]."""
    return _bolge_toplami(toplam, satir, satir, sol, sag) / (sag - sol + 1)


def _sutun_orani(toplam, sutun, ust, alt):
    """Occupancy ratio of one column over the row range [ust, alt]."""
    return _bolge_toplami(toplam, ust, alt, sutun, sutun) / (alt - ust + 1)


def _kutu_orani(toplam, ust, alt, sol, sag):
    """Occupancy ratio of the whole crop box."""
    return _bolge_toplami(toplam, ust, alt, sol, sag) / ((alt - ust + 1) * (sag - sol + 1))


def _gecerli_alani_kirp(panorama, maske=None):
    """
    Trim the black border a warp leaves around the panorama.

    An edge row or column is dropped only when it is emptier than KIRPMA_ESIGI
    *and* emptier than the box it borders. The second condition is what keeps
    the rule stable: a hole pattern that runs through the whole image (every
    third row black, one black half) is no worse at the edge than in the
    middle, so trimming it away would not improve anything and does not start.
    A hard floor of KIRPMA_EN_AZ_ORAN of each side caps the damage of any
    remaining pathological case, and the passes are bounded at two. On the
    bundled examples this lands on the same crop box as the older unbounded
    loop; what it changes is what happens on masks that loop collapsed on.

    All occupancy ratios are read off a 2-D prefix sum (integral image) built
    once, so every query costs O(1) regardless of the region size. Pass the
    union of the two validity masks as `maske`; without it the mask is guessed
    from the pixels, and dark scene content counts as border.
    """
    if maske is None:
        maske = _gecerli_maske(panorama)
    if not np.any(maske):
        return panorama

    doluluk = (maske > 0).astype(np.uint8)
    # toplam[y, x] = number of valid pixels in doluluk[:y, :x] (zero padded).
    toplam = cv2.integral(doluluk, sdepth=cv2.CV_32S)

    yukseklik, genislik = doluluk.shape
    ust, alt = 0, yukseklik - 1
    sol, sag = 0, genislik - 1
    en_az_yukseklik = max(1, int(yukseklik * KIRPMA_EN_AZ_ORAN))
    en_az_genislik = max(1, int(genislik * KIRPMA_EN_AZ_ORAN))

    def bos_mu(oran, kutu_orani):
        return oran < KIRPMA_ESIGI and oran < kutu_orani

    for _ in range(2):
        degisti = False

        while alt - ust + 1 > en_az_yukseklik:
            kutu = _kutu_orani(toplam, ust, alt, sol, sag)
            if bos_mu(_satir_orani(toplam, ust, sol, sag), kutu):
                ust += 1
            elif bos_mu(_satir_orani(toplam, alt, sol, sag), kutu):
                alt -= 1
            else:
                break
            degisti = True

        while sag - sol + 1 > en_az_genislik:
            kutu = _kutu_orani(toplam, ust, alt, sol, sag)
            if bos_mu(_sutun_orani(toplam, sol, ust, alt), kutu):
                sol += 1
            elif bos_mu(_sutun_orani(toplam, sag, ust, alt), kutu):
                sag -= 1
            else:
                break
            degisti = True

        if not degisti:
            break

    return panorama[ust : alt + 1, sol : sag + 1]


def _tam_sayi_cevirme(T):
    """Return (tx, ty) when T is a pure integer pixel translation, else None."""
    if T is None or np.asarray(T).shape != (3, 3):
        return None
    T = np.asarray(T, dtype=np.float64)
    tx, ty = T[0, 2], T[1, 2]
    beklenen = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
    if not np.array_equal(T, beklenen):
        return None
    if tx != int(tx) or ty != int(ty):
        return None
    return int(tx), int(ty)


def _cevirerek_yerlestir(goruntu, tx, ty, tuval_genislik, tuval_yukseklik):
    """Copy goruntu into a zeroed canvas at integer offset (tx, ty)."""
    tuval = np.zeros((tuval_yukseklik, tuval_genislik) + goruntu.shape[2:], dtype=goruntu.dtype)
    h, w = goruntu.shape[:2]

    # Clip to the canvas so an offset that pushes the image out never throws.
    y0, y1 = max(0, ty), min(tuval_yukseklik, ty + h)
    x0, x1 = max(0, tx), min(tuval_genislik, tx + w)
    if y1 > y0 and x1 > x0:
        tuval[y0:y1, x0:x1] = goruntu[y0 - ty : y1 - ty, x0 - tx : x1 - tx]
    return tuval


def panorama_birlestir(sol_bgr, sag_bgr, H, pozlama_dengele=True):
    """
    Stage 5 — panorama stitch:
    1. Warp the left image into the right image plane with H.
    2. Align both images on a shared canvas.
    3. Match the exposure of the two shots on their overlap.
    4. Feather-blend the overlap.
    5. Auto-crop resulting black borders.

    H maps left-image coordinates into the right-image plane. Pass
    pozlama_dengele=False to keep the left image's own exposure.
    """
    _homografiyi_dogrula(H)

    h_sol, w_sol = sol_bgr.shape[:2]
    h_sag, w_sag = sag_bgr.shape[:2]

    tuval_genislik, tuval_yukseklik, T = _tuval_ve_cevirme(H, h_sol, w_sol, h_sag, w_sag)
    M_sol = T @ H

    warp_sol = cv2.warpPerspective(
        sol_bgr, M_sol, (tuval_genislik, tuval_yukseklik), flags=cv2.INTER_CUBIC
    )
    maske_sol = _warp_maskesi(h_sol, w_sol, M_sol, tuval_genislik, tuval_yukseklik)

    # The right image is only translated by T, and by construction that
    # translation is a whole number of pixels: interpolating it would resample
    # every pixel onto itself. A slice copy gives the identical result for a
    # fraction of the cost. Fall back to warping if T is ever not integral.
    cevirme = _tam_sayi_cevirme(T)
    if cevirme is None:
        warp_sag = cv2.warpPerspective(
            sag_bgr, T, (tuval_genislik, tuval_yukseklik), flags=cv2.INTER_CUBIC
        )
        maske_sag = _warp_maskesi(h_sag, w_sag, T, tuval_genislik, tuval_yukseklik)
    else:
        warp_sag = _cevirerek_yerlestir(
            sag_bgr, cevirme[0], cevirme[1], tuval_genislik, tuval_yukseklik
        )
        maske_sag = _yerlestirme_maskesi(
            h_sag, w_sag, cevirme[0], cevirme[1], tuval_genislik, tuval_yukseklik
        )

    if pozlama_dengele:
        warp_sol = _pozlama_esitle(warp_sol, warp_sag, maske_sol, maske_sag)

    panorama = _tuy_birlestir(warp_sol, warp_sag, maske_sol, maske_sag)
    return _gecerli_alani_kirp(panorama, cv2.bitwise_or(maske_sol, maske_sag))
