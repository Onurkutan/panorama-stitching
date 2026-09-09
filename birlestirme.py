import cv2
import numpy as np

from errors import PanoramaError

# A panorama of two overlapping photos stays within a few times the input
# area. A canvas larger than this multiple of the total input pixel count
# means the homography is degenerate, so refuse it before allocating it.
MAX_TUVAL_KATSAYISI = 8


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
    """Separate valid image pixels from black background."""
    return np.any(goruntu > 0, axis=2).astype(np.uint8) * 255


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


def _gecerli_alani_kirp(panorama):
    """
    Gradually trim black borders after warping using edge occupancy ratios.
    More effective than a simple bounding box, but only removes sparse edges
    to avoid over-cropping the panorama.

    Trimming order, the 0.8 threshold and the resulting crop box are exactly
    the same as the straightforward version. The only difference is that the
    edge occupancy ratios come from a 2-D prefix sum (integral image) built
    once, so every step costs O(1) instead of re-reducing the whole remaining
    region.
    """
    maske = _gecerli_maske(panorama)
    if not np.any(maske):
        return panorama

    doluluk = (maske > 0).astype(np.uint8)
    # toplam[y, x] = number of valid pixels in doluluk[:y, :x] (zero padded).
    toplam = cv2.integral(doluluk, sdepth=cv2.CV_32S)

    ust, alt = 0, doluluk.shape[0] - 1
    sol, sag = 0, doluluk.shape[1] - 1
    esik = 0.8
    degisti = True

    while degisti and ust < alt and sol < sag:
        degisti = False
        # The plain version computed both ratio arrays once per outer pass, so
        # the column loops read their first value from the row window as it was
        # BEFORE the row loops trimmed it, and only refresh after a step. Keep
        # that behaviour by tracking which row window the column ratio reflects.
        s_ust, s_alt = ust, alt

        while ust < alt and _satir_orani(toplam, ust, sol, sag) < esik:
            ust += 1
            degisti = True

        while ust < alt and _satir_orani(toplam, alt, sol, sag) < esik:
            alt -= 1
            degisti = True

        while sol < sag and _sutun_orani(toplam, sol, s_ust, s_alt) < esik:
            sol += 1
            degisti = True
            s_ust, s_alt = ust, alt

        while sol < sag and _sutun_orani(toplam, sag, s_ust, s_alt) < esik:
            sag -= 1
            degisti = True
            s_ust, s_alt = ust, alt

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


def _tuy_birlestir(warp_sol, warp_sag):
    """
    Feather-blend two warped BGR images in a narrow horizontal overlap band.
    Blending only a subset of the overlap reduces blur on clock-like scenes.
    """
    m1 = _gecerli_maske(warp_sol)
    m2 = _gecerli_maske(warp_sag)

    sol_f = warp_sol.astype(np.float32)
    sag_f = warp_sag.astype(np.float32)
    sonuc = np.zeros_like(sol_f)

    sadece_sol = (m1 > 0) & (m2 == 0)
    sadece_sag = (m2 > 0) & (m1 == 0)
    overlap = (m1 > 0) & (m2 > 0)

    sonuc[sadece_sol] = sol_f[sadece_sol]
    sonuc[sadece_sag] = sag_f[sadece_sag]

    if np.any(overlap):
        overlap_sutunlari = np.where(np.any(overlap, axis=0))[0]
        sol_sinir = int(overlap_sutunlari[0])
        sag_sinir = int(overlap_sutunlari[-1])
        orta = 0.5 * (sol_sinir + sag_sinir)

        overlap_genisligi = max(1, sag_sinir - sol_sinir + 1)
        yari_bant = max(20, min(120, overlap_genisligi // 6))

        x_koordinatlari = np.arange(warp_sol.shape[1], dtype=np.float32)
        w_sol_1d = np.clip((orta + yari_bant - x_koordinatlari) / (2 * yari_bant), 0.0, 1.0)
        w_sag_1d = 1.0 - w_sol_1d

        w_sol = np.broadcast_to(w_sol_1d, overlap.shape)[..., None]
        w_sag = np.broadcast_to(w_sag_1d, overlap.shape)[..., None]
        overlap3 = overlap[..., None]
        blended = sol_f * w_sol + sag_f * w_sag
        sonuc = np.where(overlap3, blended, sonuc)

    return np.clip(sonuc, 0, 255).astype(np.uint8)


def panorama_birlestir(sol_bgr, sag_bgr, H):
    """
    Stage 5 — panorama stitch:
    1. Warp the left image into the right image plane with H.
    2. Align both images on a shared canvas.
    3. Feather-blend overlapping regions.
    4. Auto-crop resulting black borders.

    H maps left-image coordinates into the right-image plane.
    """
    _homografiyi_dogrula(H)

    h_sol, w_sol = sol_bgr.shape[:2]
    h_sag, w_sag = sag_bgr.shape[:2]

    tuval_genislik, tuval_yukseklik, T = _tuval_ve_cevirme(H, h_sol, w_sol, h_sag, w_sag)
    M_sol = T @ H

    warp_sol = cv2.warpPerspective(
        sol_bgr, M_sol, (tuval_genislik, tuval_yukseklik), flags=cv2.INTER_CUBIC
    )

    # The right image is only translated by T, and by construction that
    # translation is a whole number of pixels: interpolating it would resample
    # every pixel onto itself. A slice copy gives the identical result for a
    # fraction of the cost. Fall back to warping if T is ever not integral.
    cevirme = _tam_sayi_cevirme(T)
    if cevirme is None:
        warp_sag = cv2.warpPerspective(
            sag_bgr, T, (tuval_genislik, tuval_yukseklik), flags=cv2.INTER_CUBIC
        )
    else:
        warp_sag = _cevirerek_yerlestir(
            sag_bgr, cevirme[0], cevirme[1], tuval_genislik, tuval_yukseklik
        )

    panorama = _tuy_birlestir(warp_sol, warp_sag)
    panorama = _gecerli_alani_kirp(panorama)
    return panorama
