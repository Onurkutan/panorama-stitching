"""End-to-end pipeline: two overlapping photos in, one panorama out.

This is a library module: it never configures logging, the entry points
(panorama_stitching.cli, app.py) decide how log records are handled.
"""

from pathlib import Path

import cv2
import numpy as np

from . import blending, homography, matching
from .errors import PanoramaError
from .features import detect_features
from .homography import MIN_MATCH_COUNT

__all__ = [
    "EXAMPLES",
    "PROJECT_DIR",
    "PanoramaError",
    "example_paths",
    "get_example",
    "stitch_pair",
]


# The repository root: this file lives in <root>/panorama_stitching/.
PROJECT_DIR = Path(__file__).resolve().parent.parent

EXAMPLES = [
    {
        "id": "clock",
        "title": "Clock tower",
        "folder": "images/Clock",
        "left": "sol1.jpg",
        "right": "sag1.jpg",
    },
    {
        "id": "school",
        "title": "School yard",
        "folder": "images/SchoolImage",
        "left": "sol2.jpg",
        "right": "sag2.jpg",
    },
    {
        "id": "street",
        "title": "Pont du Gard",
        "folder": "images/test1",
        "left": "s1.jpg",
        "right": "s2.jpg",
    },
]


def get_example(example_id):
    """Return the EXAMPLES entry with this id, or raise PanoramaError."""
    for example in EXAMPLES:
        if example["id"] == example_id:
            return example
    raise PanoramaError("The selected example set was not found.", code="example_not_found")


def example_paths(example_id):
    """Absolute (left, right) image paths of one bundled example set."""
    example = get_example(example_id)
    folder = PROJECT_DIR / example["folder"]
    return folder / example["left"], folder / example["right"]


def _read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        # Path() so a plain string path also produces a message, not a crash.
        name = Path(path).name
        raise PanoramaError(f"Could not read the image: {name}", code="image_unreadable")
    return image


def _downscale(image, scale):
    """Downscale with INTER_AREA, the right filter for shrinking."""
    new_width = max(1, int(round(image.shape[1] * scale)))
    new_height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _scale_inputs(left, right, max_side):
    """Shrink both inputs by one common factor when either side is too large.

    Returns (left, right, scale); scale is 1.0 when nothing was resized.
    """
    if not max_side:
        return left, right, 1.0
    longest_side = max(left.shape[0], left.shape[1], right.shape[0], right.shape[1])
    if longest_side <= max_side:
        return left, right, 1.0
    scale = float(max_side) / float(longest_side)
    return _downscale(left, scale), _downscale(right, scale), scale


def _write_image(path, image):
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise PanoramaError(f"Could not save the output: {path.name}", code="output_write_failed")


def stitch_pair(left_path, right_path, output_dir, max_side=None):
    """Run the full pipeline on one image pair and write the outputs.

    max_side: optional pixel cap on the longest side of the inputs. When the
    pair is larger, both images are downscaled by the same factor before
    detection (the applied factor is reported as "inputScale").
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    left = _read_image(left_path)
    right = _read_image(right_path)
    left, right, input_scale = _scale_inputs(left, right, max_side)

    kp_left, des_left, left_keypoints = detect_features(left)
    kp_right, des_right, right_keypoints = detect_features(right)

    left_keypoints_path = output_dir / "left_keypoints.jpg"
    right_keypoints_path = output_dir / "right_keypoints.jpg"
    _write_image(left_keypoints_path, left_keypoints)
    _write_image(right_keypoints_path, right_keypoints)

    good_matches = matching.match_features(kp_left, des_left, kp_right, des_right)
    if len(good_matches) < MIN_MATCH_COUNT:
        raise PanoramaError(
            f"Not enough matches. Required: {MIN_MATCH_COUNT}, found: {len(good_matches)}",
            code="not_enough_matches",
        )

    match_preview = cv2.drawMatches(
        left,
        kp_left,
        right,
        kp_right,
        good_matches[:80],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    match_path = output_dir / "matches.jpg"
    _write_image(match_path, match_preview)

    ransac_path = output_dir / "ransac_inliers.jpg"
    H, mask = homography.estimate_homography(
        kp_left,
        kp_right,
        good_matches,
        left,
        right,
        inlier_output_path=str(ransac_path),
    )
    if H is None or mask is None:
        raise PanoramaError(
            "Could not compute the homography; the images may not overlap enough.",
            code="homography_failed",
        )

    panorama = blending.stitch_images(left, right, H)
    if panorama is None:
        raise PanoramaError("The panorama could not be produced.", code="unexpected")

    panorama_path = output_dir / "panorama.jpg"
    _write_image(panorama_path, panorama)

    inliers = int(np.sum(mask))
    return {
        "metrics": {
            "leftKeypoints": len(kp_left),
            "rightKeypoints": len(kp_right),
            "goodMatches": len(good_matches),
            "inliers": inliers,
            "panoramaWidth": int(panorama.shape[1]),
            "panoramaHeight": int(panorama.shape[0]),
            "inputScale": round(float(input_scale), 4),
        },
        "files": {
            "panorama": panorama_path.name,
            "leftKeypoints": left_keypoints_path.name,
            "rightKeypoints": right_keypoints_path.name,
            "matches": match_path.name,
            "ransac": ransac_path.name,
        },
    }
