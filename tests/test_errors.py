"""Every PanoramaError carries a stable code the front end can branch on."""

import cv2
import numpy as np
import pytest

from panorama_stitching import blending
from panorama_stitching.errors import ERROR_CODES, PanoramaError
from panorama_stitching.features import detect_features
from panorama_stitching.homography import MIN_MATCH_COUNT, estimate_homography
from panorama_stitching.matching import match_features
from panorama_stitching.pipeline import get_example, stitch_pair


def _fake_matches(n):
    matches = []
    for i in range(n):
        m = cv2.DMatch()
        m.queryIdx = i
        m.trainIdx = i
        m.imgIdx = 0
        m.distance = 0.0
        matches.append(m)
    return matches


def test_default_code_is_unexpected():
    error = PanoramaError("something went wrong")
    assert error.code == "unexpected"
    assert str(error) == "something went wrong"


def test_every_code_is_declared():
    assert PanoramaError("x", code="image_unreadable").code in ERROR_CODES
    for code in ("too_few_images", "too_many_images", "image_not_connected"):
        assert code in ERROR_CODES


def test_details_default_to_none():
    assert PanoramaError("something went wrong").details is None
    error = PanoramaError("nope", code="image_not_connected", details={"image": 3})
    assert error.details == {"image": 3}


def test_flat_image_has_not_enough_features_code():
    flat = np.full((80, 80, 3), 128, dtype=np.uint8)
    with pytest.raises(PanoramaError) as exc_info:
        detect_features(flat)
    assert exc_info.value.code == "not_enough_features"


def test_missing_descriptors_have_not_enough_features_code():
    with pytest.raises(PanoramaError) as exc_info:
        match_features(None, None, None, None)
    assert exc_info.value.code == "not_enough_features"


def test_too_few_matches_have_not_enough_matches_code():
    n = MIN_MATCH_COUNT - 1
    kp = [cv2.KeyPoint(float(i), float(i), 1) for i in range(n)]
    dummy = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(PanoramaError) as exc_info:
        estimate_homography(kp, kp, _fake_matches(n), dummy, dummy)
    assert exc_info.value.code == "not_enough_matches"


def test_degenerate_homography_code():
    # Singular: the whole image collapses onto a line.
    singular = np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(PanoramaError) as exc_info:
        blending._compute_canvas(singular, 100, 100, 100, 100)
    assert exc_info.value.code == "degenerate_homography"

    # Finite but far too large a canvas for the input pixel count.
    with pytest.raises(PanoramaError) as exc_info:
        blending._compute_canvas(np.diag([50.0, 50.0, 1.0]), 100, 100, 100, 100)
    assert exc_info.value.code == "degenerate_homography"


def test_missing_homography_code():
    with pytest.raises(PanoramaError) as exc_info:
        blending.stitch_images(
            np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.uint8), None
        )
    assert exc_info.value.code == "homography_failed"


def test_unreadable_image_code(tmp_path):
    with pytest.raises(PanoramaError) as exc_info:
        stitch_pair(tmp_path / "missing_left.jpg", tmp_path / "missing_right.jpg", tmp_path / "out")
    assert exc_info.value.code == "image_unreadable"


def test_unknown_example_code():
    with pytest.raises(PanoramaError) as exc_info:
        get_example("does-not-exist")
    assert exc_info.value.code == "example_not_found"
