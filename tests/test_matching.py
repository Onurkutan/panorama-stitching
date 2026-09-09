import cv2
import numpy as np
import pytest

from errors import PanoramaError
from matcher import match_features


def test_match_features_raises_on_missing_descriptors():
    with pytest.raises(PanoramaError):
        match_features(None, None, None, None)


def test_match_features_raises_on_single_descriptor():
    single = np.zeros((1, 128), dtype=np.float32)
    enough = np.zeros((5, 128), dtype=np.float32)
    with pytest.raises(PanoramaError):
        match_features(None, single, None, enough)
    with pytest.raises(PanoramaError):
        match_features(None, enough, None, single)


def test_match_features_noisy_self_match(sift_descriptors):
    """A descriptor set matched against a noisy copy of itself should mostly
    recover the identity correspondence (queryIdx == trainIdx)."""
    _keypoints, descriptors = sift_descriptors
    rng = np.random.default_rng(7)
    noisy = (descriptors + rng.normal(0, 1.0, size=descriptors.shape)).astype(np.float32)

    good_matches = match_features(None, descriptors, None, noisy)

    assert len(good_matches) > 0
    correct = sum(1 for m in good_matches if m.queryIdx == m.trainIdx)
    assert correct / len(good_matches) > 0.8


def test_match_features_real_overlap(overlapping_pair):
    """Two genuinely overlapping crops of the same scene should yield matches."""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(
        cv2.cvtColor(overlapping_pair["left"], cv2.COLOR_BGR2GRAY), None
    )
    kp2, des2 = sift.detectAndCompute(
        cv2.cvtColor(overlapping_pair["right"], cv2.COLOR_BGR2GRAY), None
    )
    good_matches = match_features(kp1, des1, kp2, des2)
    assert len(good_matches) > 0
