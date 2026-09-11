"""Command line entry point: stitch the bundled example sets, or your own photos.

The pipeline itself lives in panorama_stitching.pipeline (shared with the web
app); this file only picks the images, decides where the outputs go, and shows
the last panorama on screen.

    python -m panorama_stitching.cli                      # all sets -> outputs/<set>/
    python -m panorama_stitching.cli --sets clock         # one bundled set only
    python -m panorama_stitching.cli a.jpg b.jpg c.jpg    # own photos -> outputs/custom/
    python -m panorama_stitching.cli --headless           # no GUI (also: HEADLESS=1)
    python -m panorama_stitching.cli --out /tmp/panorama  # different output root
    python -m panorama_stitching.cli *.jpg --projection planar   # no cylinder
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2

from .errors import PanoramaError
from .pipeline import (
    EXAMPLES,
    MAX_IMAGES,
    MIN_IMAGES,
    PROJECT_DIR,
    PROJECTIONS,
    example_paths,
    stitch_pair,
    stitch_set,
)

SET_NAMES = [example["id"] for example in EXAMPLES]


def _headless_env():
    """True when the environment asks for a GUI-free run (HEADLESS=1)."""
    return os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "images",
        nargs="*",
        metavar="IMAGE",
        help=f"{MIN_IMAGES} to {MAX_IMAGES} overlapping photos, in any order. "
        "Without them the bundled example sets are processed.",
    )
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
        help="Subset of the bundled image sets to process (default: all); "
        "ignored when photos are given.",
    )
    parser.add_argument(
        "--name",
        default="custom",
        help="Sub-directory of --out the given photos are written to (default: custom).",
    )
    parser.add_argument(
        "--projection",
        choices=PROJECTIONS,
        default="auto",
        help="Surface the photos are projected onto: auto (default; planar for "
        "a pair, cylindrical from three photos on), planar or cylindrical. "
        "Ignored for the bundled example sets, which are pairs.",
    )
    return parser.parse_args(argv)


def _output_root(out):
    """Resolve --out against the project directory when it is relative."""
    path = Path(out)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


def _process_set(set_name, output_root, projection="auto"):
    """Stitch one example set; returns the panorama path.

    Pairs go through stitch_pair (two-image outputs), larger sets through
    stitch_set exactly like photos given on the command line.
    """
    paths = example_paths(set_name)
    if len(paths) != 2:
        return _process_images(list(paths), output_root, set_name, projection)

    left_path, right_path = paths
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


def _process_images(image_paths, output_root, name, projection="auto"):
    """Stitch one set of photos given on the command line; returns the panorama path."""
    set_output = output_root / name
    listed = ", ".join(Path(path).name for path in image_paths)
    print(f"\n=== {name} ({len(image_paths)} photos: {listed}) ===")
    result = stitch_set(image_paths, set_output, projection=projection)

    metrics = result["metrics"]
    for index, keypoint_count in enumerate(metrics["keypoints"], start=1):
        print(f"Image {index}: {keypoint_count} keypoints")
    for pair in metrics["pairs"]:
        first, second = pair["images"]
        print(
            f"Pair {first}+{second}: {pair['goodMatches']} good matches, "
            f"{pair['inliers']} RANSAC inliers"
        )
    order = " -> ".join(str(index) for index in metrics["order"])
    print(f"Reference: image {metrics['reference']} | Detected order: {order}")
    focal = metrics["focalPx"]
    requested = "" if projection == metrics["projection"] else f" (requested: {projection})"
    print(
        f"Projection: {metrics['projection']}{requested}"
        + ("" if focal is None else f" | Focal: {focal:.0f} px")
    )
    print(
        f"Panorama: {metrics['panoramaWidth']}x{metrics['panoramaHeight']} "
        f"| Scale: {metrics['inputScale']}"
    )
    panorama_path = set_output / result["files"]["panorama"]
    print(f"Saved: {panorama_path}")
    return panorama_path


def _process_examples(set_names, output_root):
    """Stitch the bundled sets; returns the last panorama that was produced."""
    last_panorama = None
    succeeded = 0
    for set_name in set_names:
        try:
            last_panorama = _process_set(set_name, output_root) or last_panorama
            succeeded += 1
        except PanoramaError as error:
            # One bad set must not stop the others.
            print(f"Skipped this set ({set_name}): {error}")

    print(f"\nDone! Processed {succeeded}/{len(set_names)} image sets.")
    return last_panorama


def main(argv=None):
    args = _parse_args(argv)
    # The entry point (not the library) decides how log records are shown.
    # stdout (not the default stderr) so log lines stay ordered with the prints.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    show_window = not (args.headless or _headless_env())
    output_root = _output_root(args.out)

    if args.images:
        try:
            last_panorama = _process_images(args.images, output_root, args.name, args.projection)
        except PanoramaError as error:
            print(f"Could not stitch these photos: {error}")
            return 1
        print("\nDone! Stitched 1 set of photos.")
    else:
        last_panorama = _process_examples(args.sets, output_root)

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
