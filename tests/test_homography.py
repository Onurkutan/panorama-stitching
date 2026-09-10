import cv2
import numpy as np
import pytest

from panorama_stitching.errors import PanoramaError
from panorama_stitching.homography import MIN_MATCH_COUNT, estimate_homography


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


def _known_homography_points():
    """A grid of points in a "left" image plus their images under a known H."""
    H = np.array(
        [[1.2, 0.05, 15.0], [-0.03, 1.1, 8.0], [0.0002, 0.0001, 1.0]],
        dtype=np.float64,
    )
    xs, ys = np.meshgrid(np.linspace(10, 190, 6), np.linspace(10, 140, 5))
    left_pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    homogeneous = np.hstack([left_pts, np.ones((len(left_pts), 1))])
    mapped = (H @ homogeneous.T).T
    mapped = mapped[:, :2] / mapped[:, 2:3]
    return H, left_pts, mapped


def test_too_few_matches_raises():
    n = MIN_MATCH_COUNT - 1
    kp = [cv2.KeyPoint(float(i), float(i), 1) for i in range(n)]
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(PanoramaError):
        estimate_homography(kp, kp, _fake_matches(n), dummy_img, dummy_img)


def test_recovers_known_homography():
    H_true, left_pts, right_pts = _known_homography_points()
    kp1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in left_pts]
    kp2 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in right_pts]
    matches = _fake_matches(len(left_pts))
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)

    H_est, mask = estimate_homography(kp1, kp2, matches, dummy_img, dummy_img)

    assert int(mask.sum()) == len(matches)
    H_true_n = H_true / H_true[2, 2]
    H_est_n = H_est / H_est[2, 2]
    assert np.allclose(H_true_n, H_est_n, atol=1e-2)


def test_no_file_written_without_output_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    H_true, left_pts, right_pts = _known_homography_points()
    kp1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in left_pts]
    kp2 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in right_pts]
    matches = _fake_matches(len(left_pts))
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)

    estimate_homography(kp1, kp2, matches, dummy_img, dummy_img, inlier_output_path=None)

    assert list(tmp_path.iterdir()) == []


def test_writes_file_when_output_path_given(tmp_path):
    H_true, left_pts, right_pts = _known_homography_points()
    kp1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in left_pts]
    kp2 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in right_pts]
    matches = _fake_matches(len(left_pts))
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    out_path = tmp_path / "ransac.jpg"

    estimate_homography(kp1, kp2, matches, dummy_img, dummy_img, inlier_output_path=str(out_path))

    assert out_path.exists()
