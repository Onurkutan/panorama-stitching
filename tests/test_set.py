"""stitch_set: 2..MAX_IMAGES photos in any order, arranged by pairwise matching."""

import hashlib
import math

import cv2
import numpy as np
import pytest

from panorama_stitching import blending, pipeline
from panorama_stitching.errors import PanoramaError
from panorama_stitching.pipeline import MAX_IMAGES, stitch_pair, stitch_set

# The order the three crops are handed over in: crop 3, crop 1, crop 2. Every
# assertion about metrics["order"] is mapped back through this, so the test
# fails if the pipeline just echoes the input order.
SHUFFLED = [2, 0, 1]

# Same idea for the four frames of the rotating sweep.
SWEEP_SHUFFLED = [2, 0, 3, 1]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _progress_total(count, projection="planar"):
    steps = count + count * (count - 1) // 2 + 3
    return steps + (count - 1) if projection == "cylindrical" else steps


def test_shuffled_crops_recover_the_arrangement(tmp_path, crop_set_paths):
    steps = []
    result = stitch_set(
        [crop_set_paths[index] for index in SHUFFLED],
        tmp_path,
        progress=lambda stage, step, total: steps.append((stage, step, total)),
        projection="planar",
    )
    metrics = result["metrics"]
    files = result["files"]
    count = len(crop_set_paths)

    assert metrics["imageCount"] == count
    assert len(metrics["keypoints"]) == count
    assert len(metrics["pairs"]) == count - 1
    assert metrics["pairsEvaluated"] == count * (count - 1) // 2
    assert metrics["totalKeypoints"] == sum(metrics["keypoints"])
    assert metrics["totalGoodMatches"] == sum(pair["goodMatches"] for pair in metrics["pairs"])
    assert metrics["totalInliers"] == sum(pair["inliers"] for pair in metrics["pairs"])
    assert 1 <= metrics["reference"] <= count
    assert metrics["projection"] == "planar"
    assert metrics["focalPx"] is None

    # Read back through SHUFFLED, the detected order has to spell out the true
    # left-to-right arrangement of the crops.
    assert [SHUFFLED[index - 1] for index in metrics["order"]] == [0, 1, 2]

    widest = max(cv2.imread(str(path)).shape[1] for path in crop_set_paths)
    assert metrics["panoramaWidth"] > widest

    assert files["panorama"] == "panorama.jpg"
    assert files["keypoints"] == [f"keypoints_{k}.jpg" for k in range(1, count + 1)]
    assert len(files["matches"]) == len(files["ransac"]) == len(files["pairs"]) == count - 1
    for pair, matches_name, ransac_name in zip(
        files["pairs"], files["matches"], files["ransac"], strict=True
    ):
        first, second = pair
        assert first < second
        assert matches_name == f"matches_{first}_{second}.jpg"
        assert ransac_name == f"ransac_{first}_{second}.jpg"
    assert [pair["images"] for pair in metrics["pairs"]] == files["pairs"]

    written = [files["panorama"], *files["keypoints"], *files["matches"], *files["ransac"]]
    for name in written:
        assert cv2.imread(str(tmp_path / name)) is not None

    total = _progress_total(count)
    assert [step[0] for step in steps] == (
        ["features"] * count
        + ["matching"] * (count * (count - 1) // 2)
        + ["homography", "blending", "saving"]
    )
    assert [step[1] for step in steps] == list(range(1, total + 1))
    assert {step[2] for step in steps} == {total}


@pytest.mark.slow
def test_two_images_match_stitch_pair_byte_for_byte(tmp_path, clock_pair_paths):
    """The set path must not change what the bundled two-image demo produces."""
    left_path, right_path = clock_pair_paths
    pair = stitch_pair(left_path, right_path, tmp_path / "pair", max_side=600)
    result = stitch_set([left_path, right_path], tmp_path / "set", max_side=600)

    assert _sha256(tmp_path / "pair" / "panorama.jpg") == _sha256(
        tmp_path / "set" / "panorama.jpg",
    )

    metrics, pair_metrics = result["metrics"], pair["metrics"]
    assert metrics["imageCount"] == 2
    assert metrics["keypoints"] == [
        pair_metrics["leftKeypoints"],
        pair_metrics["rightKeypoints"],
    ]
    assert metrics["pairs"] == [
        {
            "images": [1, 2],
            "goodMatches": pair_metrics["goodMatches"],
            "inliers": pair_metrics["inliers"],
        }
    ]
    # The tree centre of a single edge is its later node, so the second photo
    # is the reference -- the very plane stitch_pair warps the first one into.
    assert metrics["reference"] == 2
    assert metrics["order"] == [1, 2]
    # Two photos are one hop apart, so "auto" leaves the pair on its plane.
    assert metrics["projection"] == "planar"
    assert metrics["focalPx"] is None
    for key in ("panoramaWidth", "panoramaHeight", "inputScale"):
        assert metrics[key] == pair_metrics[key]


def test_stitch_set_images_matches_stitch_images(overlapping_pair):
    """The two-image case of the compositor, checked without any file I/O."""
    left, right = overlapping_pair["left"], overlapping_pair["right"]
    H = np.float32([[1, 0, -overlapping_pair["shift"]], [0, 1, 0], [0, 0, 1]])

    pair = blending.stitch_images(left, right, H)
    as_set = blending.stitch_set_images([left, right], [H, np.eye(3)], 1, [0])

    assert np.array_equal(pair, as_set)


def test_too_few_images(tmp_path, crop_set_paths):
    with pytest.raises(PanoramaError) as exc_info:
        stitch_set(crop_set_paths[:1], tmp_path)
    assert exc_info.value.code == "too_few_images"


def test_too_many_images(tmp_path, crop_set_paths):
    # The count is checked before anything is read, so repeats are free here.
    with pytest.raises(PanoramaError) as exc_info:
        stitch_set([crop_set_paths[0]] * (MAX_IMAGES + 1), tmp_path)
    assert exc_info.value.code == "too_many_images"


def test_unreadable_image_names_its_index(tmp_path, crop_set_paths):
    with pytest.raises(PanoramaError) as exc_info:
        stitch_set([crop_set_paths[0], tmp_path / "missing.jpg"], tmp_path / "out")
    assert exc_info.value.code == "image_unreadable"
    assert exc_info.value.details == {"image": 2}


def test_photo_without_any_overlap_is_reported(tmp_path, crop_set_paths, unrelated_image_path):
    """A photo of another scene matches nothing, so it cannot join the tree."""
    paths = [crop_set_paths[0], crop_set_paths[1], unrelated_image_path]
    with pytest.raises(PanoramaError) as exc_info:
        stitch_set(paths, tmp_path)
    assert exc_info.value.code == "image_not_connected"
    assert exc_info.value.details == {"image": 3}
    assert "3" in str(exc_info.value)


# ------------------------------------------------------- planar vs cylinder


def test_unknown_projection_is_refused(tmp_path, crop_set_paths):
    with pytest.raises(PanoramaError) as exc_info:
        stitch_set(crop_set_paths, tmp_path, projection="spherical")
    assert exc_info.value.code == "form_invalid"


def test_cylindrical_projection_on_the_crop_set(tmp_path, crop_set_paths):
    """Crops of one flat scene are translations, so the cylinder only bends them.

    There is no rotation to undo here and the crops carry no focal information
    either, so the projection falls back to a default cylinder and the panorama
    comes out mildly curved. What matters is that the path holds together: the
    arrangement is still recovered and the result still covers the whole scene.
    """
    count = len(crop_set_paths)
    steps = []
    result = stitch_set(
        [crop_set_paths[index] for index in SHUFFLED],
        tmp_path,
        progress=lambda stage, step, total: steps.append((stage, step, total)),
        projection="cylindrical",
    )
    metrics = result["metrics"]

    assert metrics["projection"] == "cylindrical"
    assert metrics["focalPx"] > 0
    assert [SHUFFLED[index - 1] for index in metrics["order"]] == [0, 1, 2]

    widest = max(cv2.imread(str(path)).shape[1] for path in crop_set_paths)
    assert widest < metrics["panoramaWidth"] < 3 * widest
    assert cv2.imread(str(tmp_path / result["files"]["panorama"])) is not None

    # The tree edges are matched a second time on the cylinder: N-1 steps more
    # than the planar run, all of them still reported as "matching".
    total = _progress_total(count, "cylindrical")
    assert total == _progress_total(count) + count - 1
    assert [step[0] for step in steps] == (
        ["features"] * count
        + ["matching"] * (count * (count - 1) // 2 + count - 1)
        + ["homography", "blending", "saving"]
    )
    assert [step[1] for step in steps] == list(range(1, total + 1))
    assert {step[2] for step in steps} == {total}


def test_cylinder_that_does_not_work_out_falls_back_to_planar(
    tmp_path, crop_set_paths, monkeypatch
):
    """A cylinder nobody can build costs the run its time, not its panorama."""

    def refuse(image, focal):
        raise PanoramaError("no cylinder today", code="degenerate_homography")

    monkeypatch.setattr(pipeline, "warp_cylindrical", refuse)

    steps = []
    result = stitch_set(
        crop_set_paths,
        tmp_path,
        progress=lambda stage, step, total: steps.append((stage, step, total)),
        projection="cylindrical",
    )
    metrics = result["metrics"]

    assert metrics["projection"] == "planar"
    assert metrics["focalPx"] is None
    assert cv2.imread(str(tmp_path / result["files"]["panorama"])) is not None
    # The announced total still covers the tree edges the run never reached.
    total = _progress_total(len(crop_set_paths), "cylindrical")
    assert {step[2] for step in steps} == {total}
    assert len(steps) == total - (len(crop_set_paths) - 1)


def test_rotating_sweep_needs_the_cylinder(tmp_path, rotating_sweep):
    """Ninety degrees of sweep is where the chain of planar homographies ends."""
    paths = [rotating_sweep["paths"][index] for index in SWEEP_SHUFFLED]
    width = rotating_sweep["size"][0]

    try:
        result = stitch_set(paths, tmp_path, projection="planar")
    except PanoramaError as error:
        assert error.code in {"degenerate_homography", "homography_failed"}
    else:
        # If it does come back, it is a grazing-angle smear, not a panorama:
        # the cylindrical run below covers the same sweep in far less width.
        assert result["metrics"]["panoramaWidth"] > 6 * width


@pytest.mark.parametrize("projection", ["auto", "cylindrical"])
def test_rotating_sweep_stitches_on_the_cylinder(tmp_path, rotating_sweep, projection):
    """The same four frames, projected onto a cylinder first: one clean sweep."""
    paths = [rotating_sweep["paths"][index] for index in SWEEP_SHUFFLED]
    focal = rotating_sweep["focal"]
    result = stitch_set(paths, tmp_path, projection=projection)
    metrics = result["metrics"]

    assert metrics["projection"] == "cylindrical"
    # The focal is recovered from the tree edges, not handed in.
    assert metrics["focalPx"] == pytest.approx(focal, rel=0.1)
    assert [SWEEP_SHUFFLED[index - 1] for index in metrics["order"]] == [0, 1, 2, 3]
    assert len(metrics["pairs"]) == len(paths) - 1

    # On a cylinder of radius f, an angle costs f * angle pixels of width. The
    # panorama has to cover the sweep itself plus the outer half of each end
    # frame, and anything far beyond that would be a warp running away.
    span = math.radians(rotating_sweep["span"])
    assert focal * span < metrics["panoramaWidth"] < focal * (span + math.pi / 2)
    assert cv2.imread(str(tmp_path / result["files"]["panorama"])) is not None


def test_vertical_pair_stitches(tmp_path, vertical_pair_paths):
    """The skeleton is a tree, not a left-to-right chain: stacked crops work too."""
    top_path, bottom_path = vertical_pair_paths
    # Handed over bottom first, so nothing about the arrangement is assumed.
    result = stitch_set([bottom_path, top_path], tmp_path)

    metrics = result["metrics"]
    tallest = max(cv2.imread(str(path)).shape[0] for path in vertical_pair_paths)
    assert metrics["panoramaHeight"] > tallest
    assert len(metrics["pairs"]) == 1
    assert cv2.imread(str(tmp_path / result["files"]["panorama"])) is not None


@pytest.mark.slow
def test_balcony_demo_set_stitches_cylindrically(tmp_path):
    from panorama_stitching.pipeline import example_paths

    paths = list(example_paths("balcony"))
    assert len(paths) == 6
    result = stitch_set(paths, tmp_path, max_side=500)

    metrics = result["metrics"]
    assert metrics["projection"] == "cylindrical"
    assert metrics["order"] == [1, 2, 3, 4, 5, 6]
    assert metrics["panoramaWidth"] > 2 * 375  # well beyond a single 375-px-wide input
