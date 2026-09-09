import cv2
import numpy as np


def _tuval_ve_cevirme(H, sol_yukseklik, sol_genislik, sag_yukseklik, sag_genislik, pad=2):
    """Compute canvas size and translation from warped left and fixed right image."""
    sol_kose = np.float32(
        [[0, 0], [sol_genislik, 0], [sol_genislik, sol_yukseklik], [0, sol_yukseklik]]
    ).reshape(-1, 1, 2)
    sol_donusmus = cv2.perspectiveTransform(sol_kose, H).reshape(-1, 2)

    sag_kose = np.float32(
        [[0, 0], [sag_genislik, 0], [sag_genislik, sag_yukseklik], [0, sag_yukseklik]]
    )
    tum_noktalar = np.vstack([sol_donusmus, sag_kose])

    xmin, ymin = tum_noktalar.min(axis=0)
    xmax, ymax = tum_noktalar.max(axis=0)

    tx = int(np.floor(-xmin)) + pad
    ty = int(np.floor(-ymin)) + pad
    tuval_genislik = int(np.ceil(xmax - xmin)) + 2 * pad
    tuval_yukseklik = int(np.ceil(ymax - ymin)) + 2 * pad

    T = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    return tuval_genislik, tuval_yukseklik, T


def _gecerli_maske(goruntu):
    """Separate valid image pixels from black background."""
    return np.any(goruntu > 0, axis=2).astype(np.uint8) * 255


def _gecerli_alani_kirp(panorama):
    """
    Gradually trim black borders after warping using edge occupancy ratios.
    More effective than a simple bounding box, but only removes sparse edges
    to avoid over-cropping the panorama.
    """
    maske = _gecerli_maske(panorama)
    if not np.any(maske):
        return panorama

    doluluk = maske > 0
    ust, alt = 0, doluluk.shape[0] - 1
    sol, sag = 0, doluluk.shape[1] - 1
    esik = 0.8
    degisti = True

    while degisti and ust < alt and sol < sag:
        degisti = False
        satir_oranlari = doluluk[ust:alt + 1, sol:sag + 1].mean(axis=1)
        sutun_oranlari = doluluk[ust:alt + 1, sol:sag + 1].mean(axis=0)

        while ust < alt and satir_oranlari[0] < esik:
            ust += 1
            degisti = True
            satir_oranlari = doluluk[ust:alt + 1, sol:sag + 1].mean(axis=1)

        while ust < alt and satir_oranlari[-1] < esik:
            alt -= 1
            degisti = True
            satir_oranlari = doluluk[ust:alt + 1, sol:sag + 1].mean(axis=1)

        while sol < sag and sutun_oranlari[0] < esik:
            sol += 1
            degisti = True
            sutun_oranlari = doluluk[ust:alt + 1, sol:sag + 1].mean(axis=0)

        while sol < sag and sutun_oranlari[-1] < esik:
            sag -= 1
            degisti = True
            sutun_oranlari = doluluk[ust:alt + 1, sol:sag + 1].mean(axis=0)

    return panorama[ust:alt + 1, sol:sag + 1]


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
    if H is None:
        print("HATA: Homografi yok, birlestirme atlandi.")
        return None

    h_sol, w_sol = sol_bgr.shape[:2]
    h_sag, w_sag = sag_bgr.shape[:2]

    tuval_genislik, tuval_yukseklik, T = _tuval_ve_cevirme(
        H, h_sol, w_sol, h_sag, w_sag
    )
    M_sol = T @ H
    M_sag = T

    warp_sol = cv2.warpPerspective(
        sol_bgr, M_sol, (tuval_genislik, tuval_yukseklik), flags=cv2.INTER_CUBIC
    )
    warp_sag = cv2.warpPerspective(
        sag_bgr, M_sag, (tuval_genislik, tuval_yukseklik), flags=cv2.INTER_CUBIC
    )

    panorama = _tuy_birlestir(warp_sol, warp_sag)
    panorama = _gecerli_alani_kirp(panorama)
    return panorama
