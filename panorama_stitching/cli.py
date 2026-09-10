"""Command line entry point: stitch the bundled example image sets.

The pipeline itself lives in panorama_stitching.pipeline (shared with the web
app); this file only picks the image sets, decides where the outputs go, and
shows the last panorama on screen.

    python -m panorama_stitching.cli                      # all sets -> outputs/<set>/
    python -m panorama_stitching.cli --sets clock         # one set only
    python -m panorama_stitching.cli --headless           # no GUI (also: HEADLESS=1)
    python -m panorama_stitching.cli --out /tmp/panorama  # different output root
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2

from .errors import PanoramaError
from .pipeline import EXAMPLES, PROJECT_DIR, example_paths, stitch_pair

SET_NAMES = [example["id"] for example in EXAMPLES]


def _headless_env():
    """True when the environment asks for a GUI-free run (HEADLESS=1)."""
    return os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default="outputs",
        help="Output root directory; each set writes into <out>/<set>/ "
        "(relative paths are resolved against the project directory).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open any GUI window (env HEADLESS=1 does the same).",
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        choices=SET_NAMES,
        default=SET_NAMES,
        help="Subset of image sets to process (default: all).",
    )
    return parser.parse_args(argv)


def _output_root(out):
    """Resolve --out against the project directory when it is relative."""
    path = Path(out)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


def _process_set(set_name, output_root):
    """Stitch one example set; returns the panorama path."""
    left_path, right_path = example_paths(set_name)
    set_output = output_root / set_name

    print(f"\n=== {set_name} ({left_path.name} + {right_path.name}) ===")
    result = stitch_pair(left_path, right_path, set_output)

    metrics = result["metrics"]
    print(
        "Left keypoints: {leftKeypoints} | Right keypoints: {rightKeypoints} | "
        "Good matches: {goodMatches} | RANSAC inliers: {inliers} | "
        "Panorama: {panoramaWidth}x{panoramaHeight} | Scale: {inputScale}".format(**metrics)
    )
    panorama_path = set_output / result["files"]["panorama"]
    print(f"Saved: {panorama_path}")
    return panorama_path


def main(argv=None):
    args = _parse_args(argv)
    # The entry point (not the library) decides how log records are shown.
    # stdout (not the default stderr) so log lines stay ordered with the prints.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    show_window = not (args.headless or _headless_env())
    output_root = _output_root(args.out)

    last_panorama = None
    succeeded = 0
    for set_name in args.sets:
        try:
            last_panorama = _process_set(set_name, output_root) or last_panorama
            succeeded += 1
        except PanoramaError as error:
            # One bad set must not stop the others.
            print(f"Skipped this set ({set_name}): {error}")

    print(f"\nDone! Processed {succeeded}/{len(args.sets)} image sets.")

    # Show only the last panorama to avoid window clutter.
    if show_window and last_panorama is not None:
        panorama = cv2.imread(str(last_panorama))
        if panorama is None:
            print(f"Could not display the panorama: {last_panorama}")
            return 0
        cv2.imshow("Panorama (stage 5)", panorama)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
