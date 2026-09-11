"""The cylinder a wide sweep is stitched on: focal estimation and the warp."""

import math

import numpy as np
import pytest

from panorama_stitching.errors import PanoramaError
from panorama_stitching.projection import (
    FOCAL_FALLBACK_FACTOR,
    cylindrical_width,
    estimate_focal,
    warp_cylindrical,
)

WIDTH, HEIGHT = 1800, 2400
FOCAL = 1733.0  # a 26 mm equivalent phone lens at this size


def _intrinsics(focal=FOCAL, width=WIDTH, height=HEIGHT):
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])


def _yaw(degrees):
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])


def _pitch(degrees):
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])


def _rotation_homography(R, focal=FOCAL):
    """H = K R K^-1: what two shots from one turning camera are related by."""
    K = _intrinsics(focal)
    return K @ R @ np.linalg.inv(K)


def test_estimate_focal_recovers_a_known_rotation():
    """The whole point: the focal falls out of the homography alone."""
    H = _rotation_homography(_yaw(20.0))
    assert estimate_focal([H], WIDTH, HEIGHT) == pytest.approx(FOCAL, rel=0.01)


@pytest.mark.parametrize("degrees", [5.0, 15.0, 30.0, 45.0])
def test_estimate_focal_over_a_range_of_angles(degrees):
    H = _rotation_homography(_yaw(degrees))
    assert estimate_focal([H], WIDTH, HEIGHT) == pytest.approx(FOCAL, rel=0.02)


def test_estimate_focal_survives_a_handheld_tilt():
    """A sweep by hand is never a clean yaw; a few degrees of pitch must not matter."""
    H = _rotation_homography(_yaw(22.0) @ _pitch(3.0))
    assert estimate_focal([H], WIDTH, HEIGHT) == pytest.approx(FOCAL, rel=0.02)


def test_estimate_focal_takes_the_median_over_the_edges():
    """One pair that came out wrong must not drag the whole set with it."""
    good = [_rotation_homography(_yaw(degrees)) for degrees in (18.0, 24.0, 27.0)]
    nonsense = np.array([[2.0, 0.1, 30.0], [0.05, 2.0, -12.0], [1e-3, 9e-4, 1.0]])
    assert estimate_focal([*good, nonsense], WIDTH, HEIGHT) == pytest.approx(FOCAL, rel=0.02)


def test_estimate_focal_falls_back_on_a_translation():
    """A pure shift carries no focal information at all."""
    translation = np.array([[1.0, 0.0, 137.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]])
    expected = FOCAL_FALLBACK_FACTOR * HEIGHT
    assert estimate_focal([translation], WIDTH, HEIGHT) == pytest.approx(expected)


def test_estimate_focal_falls_back_without_any_homography():
    assert estimate_focal([], WIDTH, HEIGHT) == pytest.approx(FOCAL_FALLBACK_FACTOR * HEIGHT)


def test_estimate_focal_ignores_broken_matrices():
    broken = [None, np.eye(2), np.full((3, 3), np.nan)]
    assert estimate_focal(broken, WIDTH, HEIGHT) == pytest.approx(FOCAL_FALLBACK_FACTOR * HEIGHT)


def test_estimate_focal_rejects_an_implausible_estimate():
    """A focal a hundred times the frame is a degenerate homography talking."""
    H = _rotation_homography(_yaw(20.0), focal=200.0 * WIDTH)
    assert estimate_focal([H], WIDTH, HEIGHT) == pytest.approx(FOCAL_FALLBACK_FACTOR * HEIGHT)


# ------------------------------------------------------------------ the warp


@pytest.fixture(scope="module")
def textured():
    """A small odd-sized image: an odd width has an exact centre column."""
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, size=(301, 401, 3), dtype=np.uint8)


def test_warp_cylindrical_keeps_the_centre_column(textured):
    """The ray straight ahead hits the cylinder where it touches the image plane."""
    warped, _mask = warp_cylindrical(textured, 350.0)

    assert warped.shape[0] == textured.shape[0]
    assert warped.shape[1] % 2 == 1
    centre = np.array_equal(warped[:, warped.shape[1] // 2], textured[:, textured.shape[1] // 2])
    assert centre


def test_warp_cylindrical_width_follows_the_arc(textured):
    """The image spans 2 atan(w / 2f) radians, f times that in arc length."""
    focal = 350.0
    warped, _mask = warp_cylindrical(textured, focal)
    arc = focal * 2.0 * math.atan(textured.shape[1] / (2.0 * focal))
    assert warped.shape[1] == pytest.approx(arc, abs=3)
    # A longer lens is a flatter cylinder, so less of the width is folded away.
    assert warp_cylindrical(textured, 4 * focal)[0].shape[1] > warped.shape[1]


def test_warp_cylindrical_mask_describes_the_output(textured):
    warped, mask = warp_cylindrical(textured, 350.0)

    assert mask.shape == warped.shape[:2]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}
    # Cropped to the mask's bounding box: every edge still carries content.
    assert mask[0].any() and mask[-1].any()
    assert mask[:, 0].any() and mask[:, -1].any()
    # And it is a real mask, not a full rectangle: the corners are folded away.
    assert not mask.all()
    assert mask[0, 0] == 0


def test_warp_cylindrical_mask_and_image_agree(textured):
    """Nothing outside the mask carries content worth blending."""
    _warped, mask = warp_cylindrical(textured, 350.0)
    columns = mask.sum(axis=0)
    # The cylinder keeps the full height at its centre and folds the sides in.
    assert columns[mask.shape[1] // 2] == 255 * mask.shape[0]
    assert columns[0] < columns[mask.shape[1] // 2]


def test_cylindrical_width_is_odd():
    for focal in (120.0, 350.0, 1733.0):
        assert cylindrical_width(1800, focal) % 2 == 1


@pytest.mark.parametrize("focal", [0.0, -12.0, float("nan"), float("inf")])
def test_warp_cylindrical_rejects_an_unusable_focal(textured, focal):
    with pytest.raises(PanoramaError) as exc_info:
        warp_cylindrical(textured, focal)
    assert exc_info.value.code == "degenerate_homography"
