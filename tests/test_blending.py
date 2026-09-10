import tracemalloc

import cv2
import numpy as np
import pytest

import birlestirme
from errors import PanoramaError


def test_tuval_ve_cevirme_identity():
    h_sol, w_sol = 100, 150
    h_sag, w_sag = 120, 90
    pad = 2

    w, h, T = birlestirme._tuval_ve_cevirme(np.eye(3), h_sol, w_sol, h_sag, w_sag, pad=pad)

    assert w == max(w_sol, w_sag) + 2 * pad
    assert h == max(h_sol, h_sag) + 2 * pad
    tx, ty = T[0, 2], T[1, 2]
    assert tx == pad and ty == pad
    assert np.array_equal(T, np.float32([[1, 0, pad], [0, 1, pad], [0, 0, 1]]))


def test_tuval_ve_cevirme_extreme_scale_raises():
    # Not singular (det = 2500), but the resulting canvas is far larger than
    # 8x the input pixel count, so the area guard must reject it.
    H = np.diag([50.0, 50.0, 1.0])
    with pytest.raises(PanoramaError):
        birlestirme._tuval_ve_cevirme(H, 100, 100, 100, 100)


def test_tuval_ve_cevirme_nan_raises():
    H = np.eye(3)
    H[0, 0] = np.nan
    with pytest.raises(PanoramaError):
        birlestirme._tuval_ve_cevirme(H, 100, 100, 100, 100)


def test_tuval_ve_cevirme_none_raises():
    with pytest.raises(PanoramaError):
        birlestirme._tuval_ve_cevirme(None, 100, 100, 100, 100)


def test_gecerli_alani_kirp_all_black_returns_same_object():
    img = np.zeros((30, 40, 3), dtype=np.uint8)
    result = birlestirme._gecerli_alani_kirp(img)
    assert result is img


def test_gecerli_alani_kirp_l_shaped_border():
    h, w = 50, 60
    r0, c0 = 8, 10  # top and left border kept under the 20% occupancy guard
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[r0:, c0:] = 150

    result = birlestirme._gecerli_alani_kirp(img)

    assert result.shape[:2] == (h - r0, w - c0)
    assert np.array_equal(result, img[r0:, c0:])


def test_gecerli_alani_kirp_fully_valid_unchanged():
    rng = np.random.default_rng(3)
    img = rng.integers(1, 256, size=(20, 25, 3), dtype=np.uint8)
    result = birlestirme._gecerli_alani_kirp(img)
    assert result.shape == img.shape
    assert np.array_equal(result, img)


def test_tuy_birlestir():
    h, w = 50, 200
    color_sol = np.array([200, 50, 50], dtype=np.uint8)
    color_sag = np.array([50, 50, 200], dtype=np.uint8)

    warp_sol = np.zeros((h, w, 3), dtype=np.uint8)
    warp_sol[:, 0:120] = color_sol

    warp_sag = np.zeros((h, w, 3), dtype=np.uint8)
    warp_sag[:, 80:200] = color_sag

    result = birlestirme._tuy_birlestir(warp_sol, warp_sag)

    assert result.dtype == np.uint8
    assert np.array_equal(result[:, 0:80], np.broadcast_to(color_sol, (h, 80, 3)))
    assert np.array_equal(result[:, 120:200], np.broadcast_to(color_sag, (h, 80, 3)))

    centre_col = 99  # ~ midpoint of the overlap columns [80, 119]
    expected_mean = (color_sol.astype(np.float32) + color_sag.astype(np.float32)) / 2
    assert np.allclose(result[:, centre_col].astype(np.float32), expected_mean, atol=5)


def test_cevirerek_yerlestir_matches_warp_perspective():
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
    tx, ty = 15, 8
    canvas_w, canvas_h = 80, 60

    result = birlestirme._cevirerek_yerlestir(img, tx, ty, canvas_w, canvas_h)

    T = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    expected = cv2.warpPerspective(img, T, (canvas_w, canvas_h))

    assert np.array_equal(result, expected)


# --------------------------------------------------------------- validity masks


def _egik_homografi():
    """Rotation plus a little perspective: the warped border then runs
    diagonally across the canvas and interpolation has a black background to
    mix into the border pixels."""
    return np.float32([[0.92, -0.18, 120.0], [0.10, 0.95, 20.0], [1e-5, 2e-5, 1.0]])


def test_warp_maskesi_karanlik_seridi_disarida_birakir():
    duz = np.full((300, 400, 3), 200, dtype=np.uint8)
    H = _egik_homografi()
    genislik, yukseklik, T = birlestirme._tuval_ve_cevirme(H, 300, 400, 300, 400)
    M = T @ H
    warp = cv2.warpPerspective(duz, M, (genislik, yukseklik), flags=cv2.INTER_CUBIC)

    maske = birlestirme._warp_maskesi(300, 400, M, genislik, yukseklik)
    gecerli = maske > 0
    assert np.count_nonzero(gecerli) > 90000
    # Every pixel the mask calls valid is real content, not a dark fringe.
    assert warp[gecerli].min() >= 190

    # The plain "any channel > 0" rule does let the fringe through, which is
    # exactly why the warped mask exists.
    kaba = birlestirme._gecerli_maske(warp) > 0
    assert warp[kaba].min() < 60
    assert np.count_nonzero(kaba) > np.count_nonzero(gecerli)


def test_warp_maskesi_kenardan_asindirir():
    yaricap = birlestirme.MASKE_ASINDIRMA
    # A 40x60 image translated to (5, 4) on a 60x80 canvas: the whole border of
    # the placed rectangle is interior to the canvas, so all of it is eroded.
    M = np.float32([[1, 0, 5], [0, 1, 4], [0, 0, 1]])
    maske = birlestirme._warp_maskesi(40, 60, M, 80, 60)

    beklenen = np.zeros((60, 80), dtype=np.uint8)
    beklenen[4 + yaricap : 44 - yaricap, 5 + yaricap : 65 - yaricap] = 255
    assert np.array_equal(maske, beklenen)


def test_warp_maskesi_kucuk_goruntuyu_silmez():
    # Eroding a 3x3 image by 2 px would leave nothing; the mask must survive.
    assert np.count_nonzero(birlestirme._warp_maskesi(3, 3, np.eye(3), 3, 3)) > 0


def test_yerlestirme_maskesi_tam_dikdortgen():
    maske = birlestirme._yerlestirme_maskesi(20, 30, 5, 4, 60, 40)

    assert np.count_nonzero(maske) == 20 * 30
    assert np.array_equal(maske[4:24, 5:35], np.full((20, 30), 255, dtype=np.uint8))


def test_tuy_birlestir_verilen_maskeyle_siyah_icerigi_korur():
    h, w = 50, 200
    renk_sol = np.array([180, 180, 180], dtype=np.uint8)
    renk_sag = np.array([200, 50, 50], dtype=np.uint8)

    warp_sol = np.zeros((h, w, 3), dtype=np.uint8)
    warp_sol[:, 0:120] = renk_sol
    warp_sol[10:20, 95:105] = 0  # genuinely black scene content, inside the overlap
    warp_sag = np.zeros((h, w, 3), dtype=np.uint8)
    warp_sag[:, 80:200] = renk_sag

    maske_sol = np.zeros((h, w), dtype=np.uint8)
    maske_sol[:, 0:120] = 255
    maske_sag = np.zeros((h, w), dtype=np.uint8)
    maske_sag[:, 80:200] = 255

    maskeli = birlestirme._tuy_birlestir(warp_sol, warp_sag, maske_sol, maske_sag)
    maskesiz = birlestirme._tuy_birlestir(warp_sol, warp_sag)

    # With a real mask the black patch is blended like any other pixel and
    # darkens the result; the "> 0" fallback drops it and keeps the right image.
    assert maskeli[10:20, 95:105].mean() < 0.75 * float(renk_sag.mean())
    assert np.array_equal(maskesiz[10:20, 95:105], np.broadcast_to(renk_sag, (10, 10, 3)))


# ------------------------------------------------------------- feather weights


def test_tuy_birlestir_sol_goruntu_sagda_oldugunda_yonu_cevirir():
    h, w = 50, 200
    renk_sol = np.array([50, 50, 200], dtype=np.uint8)
    renk_sag = np.array([200, 50, 50], dtype=np.uint8)

    # The image passed as "sol" landed on the RIGHT half of the canvas.
    warp_sol = np.zeros((h, w, 3), dtype=np.uint8)
    warp_sol[:, 80:200] = renk_sol
    warp_sag = np.zeros((h, w, 3), dtype=np.uint8)
    warp_sag[:, 0:120] = renk_sag

    sonuc = birlestirme._tuy_birlestir(warp_sol, warp_sag)

    assert np.array_equal(sonuc[:, 0:80], np.broadcast_to(renk_sag, (h, 80, 3)))
    assert np.array_equal(sonuc[:, 120:200], np.broadcast_to(renk_sol, (h, 80, 3)))
    # Each source fades out where it ends, whichever side it landed on: at the
    # first overlap column the result still belongs to the right image.
    assert np.allclose(sonuc[:, 80].astype(np.float32), renk_sag, atol=12)
    assert np.allclose(sonuc[:, 119].astype(np.float32), renk_sol, atol=12)
    beklenen = (renk_sol.astype(np.float32) + renk_sag.astype(np.float32)) / 2
    assert np.allclose(sonuc[:, 99].astype(np.float32), beklenen, atol=8)


def test_tuy_birlestir_ortusme_kenarlarinda_kaynagina_yaklasir():
    h, w = 50, 200
    renk_sol = np.array([200, 50, 50], dtype=np.uint8)
    renk_sag = np.array([50, 50, 200], dtype=np.uint8)
    warp_sol = np.zeros((h, w, 3), dtype=np.uint8)
    warp_sol[:, 0:120] = renk_sol
    warp_sag = np.zeros((h, w, 3), dtype=np.uint8)
    warp_sag[:, 80:200] = renk_sag

    sonuc = birlestirme._tuy_birlestir(warp_sol, warp_sag)

    # No step at either end of the overlap: that step is the visible seam.
    assert np.allclose(sonuc[:, 80].astype(np.float32), renk_sol, atol=12)
    assert np.allclose(sonuc[:, 119].astype(np.float32), renk_sag, atol=12)


# ------------------------------------------------------- exposure compensation


def _kademeli(yukseklik, genislik, taban=120):
    """A smooth mid-grey ramp: bright enough to scale without clipping."""
    x = np.linspace(taban - 20, taban + 20, genislik, dtype=np.float32)
    return np.repeat(np.repeat(x[None, :, None], yukseklik, axis=0), 3, axis=2).astype(np.uint8)


def test_pozlama_esitle_ortusme_ortalamalarini_hizalar():
    warp_sol = _kademeli(40, 60)
    warp_sag = np.clip(warp_sol.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
    maske = np.full((40, 60), 255, dtype=np.uint8)

    duzeltilmis = birlestirme._pozlama_esitle(warp_sol, warp_sag, maske, maske)

    assert abs(cv2.mean(duzeltilmis, mask=maske)[0] - cv2.mean(warp_sag, mask=maske)[0]) < 1.0
    assert duzeltilmis.mean() > warp_sol.mean()


def test_pozlama_esitle_kazanci_sinirlar():
    warp_sol = np.full((20, 20, 3), 40, dtype=np.uint8)
    warp_sag = np.full((20, 20, 3), 200, dtype=np.uint8)  # ratio 5.0
    maske = np.full((20, 20), 255, dtype=np.uint8)

    duzeltilmis = birlestirme._pozlama_esitle(warp_sol, warp_sag, maske, maske)

    assert np.all(duzeltilmis == round(40 * birlestirme.POZLAMA_UST_SINIR))


def test_pozlama_esitle_ortusme_yoksa_dokunmaz():
    warp_sol = np.full((20, 40, 3), 100, dtype=np.uint8)
    warp_sag = np.full((20, 40, 3), 200, dtype=np.uint8)
    maske_sol = np.zeros((20, 40), dtype=np.uint8)
    maske_sol[:, :15] = 255
    maske_sag = np.zeros((20, 40), dtype=np.uint8)
    maske_sag[:, 25:] = 255

    assert birlestirme._pozlama_esitle(warp_sol, warp_sag, maske_sol, maske_sag) is warp_sol


def test_panorama_birlestir_pozlamayi_dengeler():
    rng = np.random.default_rng(11)
    sahne = np.full((120, 300, 3), 60, dtype=np.uint8)
    for _ in range(120):
        cv2.circle(
            sahne,
            (int(rng.integers(0, 300)), int(rng.integers(0, 120))),
            int(rng.integers(4, 18)),
            tuple(int(c) for c in rng.integers(60, 200, size=3)),
            -1,
        )
    kaydirma = 120
    sol = sahne[:, 0:200].copy()
    sag = np.clip(sahne[:, kaydirma:300].astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
    H = np.float32([[1, 0, -kaydirma], [0, 1, 0], [0, 0, 1]])

    acik = birlestirme.panorama_birlestir(sol, sag, H)
    kapali = birlestirme.panorama_birlestir(sol, sag, H, pozlama_dengele=False)

    assert acik.shape == kapali.shape
    assert acik.mean() > kapali.mean() * 1.03


# ------------------------------------------------------------------- auto-crop


def test_gecerli_alani_kirp_yarim_siyah_karede_cokmez():
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    img[:, :30] = 200

    sonuc = birlestirme._gecerli_alani_kirp(img)

    # The empty half goes, the full half is kept whole.
    assert sonuc.shape[:2] == (40, 30)
    assert np.array_equal(sonuc, img[:, :30])


def test_gecerli_alani_kirp_seyrek_satirlarda_cokmez():
    img = np.full((40, 60, 3), 200, dtype=np.uint8)
    img[::3] = 0

    sonuc = birlestirme._gecerli_alani_kirp(img)

    # Only the empty first and last rows go: a hole pattern that runs through
    # the whole image is no worse at the edge than in the middle, so trimming
    # the edge would not improve the box.
    assert sonuc.shape[:2] == (38, 60)


def test_gecerli_alani_kirp_maske_ile_karanlik_icerigi_korur():
    img = np.full((40, 60, 3), 30, dtype=np.uint8)
    img[:, :10] = 0  # black, but real scene content
    maske = np.full((40, 60), 255, dtype=np.uint8)

    assert birlestirme._gecerli_alani_kirp(img, maske).shape == img.shape
    # Without the mask the black block is read as border and trimmed away.
    assert birlestirme._gecerli_alani_kirp(img).shape[:2] == (40, 50)


def test_gecerli_alani_kirp_boyut_tabanini_asmaz():
    # Pathological input: occupancy grows column by column, so every edge is
    # emptier than the box behind it and a purely greedy rule keeps eating.
    doluluk = np.zeros((40, 60), dtype=np.uint8)
    for sutun in range(60):
        doluluk[: int(40 * (sutun + 1) / 60), sutun] = 255
    img = np.dstack([doluluk] * 3)

    sonuc = birlestirme._gecerli_alani_kirp(img, doluluk)

    assert sonuc.shape[0] >= 40 * birlestirme.KIRPMA_EN_AZ_ORAN
    assert sonuc.shape[1] >= 60 * birlestirme.KIRPMA_EN_AZ_ORAN


# ---------------------------------------------------------------------- memory


def test_panorama_birlestir_tuval_boyu_float_ayirmaz():
    rng = np.random.default_rng(13)
    sol = rng.integers(0, 256, size=(400, 500, 3), dtype=np.uint8)
    sag = rng.integers(0, 256, size=(400, 500, 3), dtype=np.uint8)
    H = np.float32([[1, 0, -380], [0, 1, 0], [0, 0, 1]])

    genislik, yukseklik, _ = birlestirme._tuval_ve_cevirme(H, 400, 500, 400, 500)
    tuval_baytlari = genislik * yukseklik * 3

    birlestirme.panorama_birlestir(sol, sag, H)  # warm the allocator caches up
    tracemalloc.start()
    tracemalloc.reset_peak()
    birlestirme.panorama_birlestir(sol, sag, H)
    _, tepe = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Blending the whole canvas in float32 costs ~12 bytes per pixel per
    # temporary; keeping the float work inside the overlap box holds the peak
    # near a handful of uint8 canvases. (numpy reports its allocations to
    # tracemalloc, so this measures the arrays that matter here.)
    assert tepe < 8 * tuval_baytlari


def test_mesafe_agirligi_dikis_bandi_disinda_tek_kaynak():
    # Two 200-px wide rectangles overlapping by 100 px: the seam is the
    # overlap's middle column, and beyond the band each side is pure.
    yukseklik, genislik = 60, 300
    maske_sol = np.zeros((yukseklik, genislik), np.uint8)
    maske_sag = np.zeros((yukseklik, genislik), np.uint8)
    maske_sol[:, :200] = 255
    maske_sag[:, 100:] = 255
    kutu = (0, yukseklik, 100, 200)

    agirlik = birlestirme._mesafe_agirligi(maske_sol, maske_sag, kutu)
    yari_bant = birlestirme._dikis_yari_banti(100)

    assert agirlik.shape == (yukseklik, 100)
    orta = agirlik[yukseklik // 2]
    assert abs(orta[50] - 0.5) < 0.05
    assert np.all(orta[: 50 - yari_bant - 1] == 1.0)
    assert np.all(orta[50 + yari_bant + 1 :] == 0.0)
    assert np.all(np.diff(orta) <= 0)
