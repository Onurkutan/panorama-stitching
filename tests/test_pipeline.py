import cv2
import numpy as np
import pytest

from errors import PanoramaError
from homografi import MIN_MATCH_COUNT
from panorama_pipeline import stitch_pair


@pytest.mark.slow
def test_stitch_pair_clock_example(tmp_path, clock_pair_paths):
    left_path, right_path = clock_pair_paths
    result = stitch_pair(left_path, right_path, tmp_path, max_side=600)

    metrics = result["metrics"]
    files = result["files"]

    for filename in files.values():
        out_path = tmp_path / filename
        assert out_path.exists()
        assert cv2.imread(str(out_path)) is not None

    assert metrics["inliers"] >= MIN_MATCH_COUNT
    assert metrics["inputScale"] < 1

    left_img = cv2.imread(str(left_path))
    right_img = cv2.imread(str(right_path))
    scale = metrics["inputScale"]
    # The panorama covers both scaled inputs plus their offset, so it must be
    # wider than either one individually.
    assert metrics["panoramaWidth"] > left_img.shape[1] * scale * 0.9
    assert metrics["panoramaWidth"] > right_img.shape[1] * scale * 0.9


@pytest.mark.slow
def test_stitch_pair_is_deterministic(tmp_path, clock_pair_paths):
    left_path, right_path = clock_pair_paths
    result1 = stitch_pair(left_path, right_path, tmp_path / "run1", max_side=600)
    result2 = stitch_pair(left_path, right_path, tmp_path / "run2", max_side=600)
    assert result1["metrics"] == result2["metrics"]


def test_stitch_pair_flat_images_raise(tmp_path):
    left_path = tmp_path / "left.jpg"
    right_path = tmp_path / "right.jpg"
    flat = np.full((120, 160, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(left_path), flat)
    cv2.imwrite(str(right_path), flat)

    with pytest.raises(PanoramaError):
        stitch_pair(left_path, right_path, tmp_path / "out")


def test_stitch_pair_missing_file_raises(tmp_path):
    with pytest.raises(PanoramaError):
        stitch_pair(tmp_path / "missing_left.jpg", tmp_path / "missing_right.jpg", tmp_path / "out")
