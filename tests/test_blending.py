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
