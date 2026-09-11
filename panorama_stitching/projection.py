"""Cylindrical projection -- focal length estimation and the cylinder warp.

A homography is the right model only while the photos stay close to one plane.
Across a wide sweep they do not: with 25-30 degrees between neighbours the
outer photos meet the reference plane at a grazing angle, and the chained
homography stretches them without bound (six 1800x2400 phone frames of a 150
degree sweep warp to 18631x19036 px three hops out, which is nothing a canvas
can hold).

Projecting every photo onto a cylinder of radius f first removes that growth.
A camera that only rotates about its vertical axis sees the very same cylinder
from every shot, so two cylindrical images differ by a rotation about the
cylinder axis -- a translation along x, plus whatever small tilt and roll the
hand added. Those chain over any number of hops without exploding, which is
exactly what stitch_set needs for three photos or more.

The one unknown a cylinder needs is the focal length in pixels. EXIF carries
it, but the browser build only ever sees decoded pixels, so it is recovered
from the homographies the matching stage has already produced (Szeliski's
construction, the same one OpenCV's stitching module uses in
`focalsFromHomography`).
"""

import math

import cv2
import numpy as np

from .errors import PanoramaError

__all__ = ["estimate_focal", "focal_candidates", "warp_cylindrical"]

# Focal used when no homography yields a usable estimate: 0.7 times the longest
# side is about a 71 degree horizontal field of view, the middle of the range
# phone and compact cameras actually cover.
FOCAL_FALLBACK_FACTOR = 0.7

# An estimate outside this range of the longest side is not a focal length but
# the echo of a degenerate homography, and one outlier would drag a median of
# two or three values with it. A 0.25x factor is a 127 degree field of view (a
# very wide lens), a 12x factor a long tele -- neither is what this pipeline
# sees, so anything beyond them is dropped before the median is taken.
FOCAL_MIN_FACTOR = 0.25
FOCAL_MAX_FACTOR = 12.0


def _to_centred(H, width, height):
    """Re-express a pixel-coordinate homography around the image centre.

    The estimates below assume a camera matrix K = diag(f, f, 1), i.e. that the
    principal point sits at the origin, while the homographies the pipeline
    computes map pixel coordinates with the origin in the top-left corner.
    H_centred = C^-1 H C, with C the translation by the principal point, moves
    between the two (both images of a pair are assumed to be the same size,
    which is what a sweep from one camera gives).
    """
    cx, cy = width / 2.0, height / 2.0
    to_pixels = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
    to_centred = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    return to_centred @ np.asarray(H, dtype=np.float64) @ to_pixels


def _pick_focal(v1, v2, d1, d2):
    """Turn a pair of focal-square candidates into one focal, or None.

    Both candidates estimate the same f squared and normally agree closely;
    when they both hold, the larger denominator decides which one is kept,
    exactly as OpenCV does it. A non-finite candidate (a zero denominator, say)
    is pushed below zero so it is simply not chosen, rather than poisoning the
    comparisons with a NaN.
    """
    v1 = v1 if np.isfinite(v1) else -np.inf
    v2 = v2 if np.isfinite(v2) else -np.inf
    if v1 < v2:
        v1, v2 = v2, v1
    if v1 > 0 and v2 > 0:
        return math.sqrt(v1 if abs(d1) > abs(d2) else v2)
    if v1 > 0:
        return math.sqrt(v1)
    return None


def focal_candidates(H):
    """The two focal lengths a homography between equal-focal views implies.

    H must already be expressed around the principal point. Writing
    H = K2 R K1^-1 with K1 = K2 = diag(f, f, 1) and R a rotation, orthonormality
    of R turns into two independent equations per image -- one on the last row
    of H (the focal of the first view), one on the last column (the focal of the
    second). Returns a tuple of the estimates that came out real and positive;
    a pure translation or a badly conditioned H yields none of them.

    This is Szeliski's derivation as OpenCV implements it in
    `cv::detail::focalsFromHomography`.
    """
    h = np.asarray(H, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(h)):
        return ()

    with np.errstate(divide="ignore", invalid="ignore"):
        # Focal of the second view, from the last row of H.
        d1 = h[6] * h[7]
        d2 = (h[7] - h[6]) * (h[7] + h[6])
        v1 = -(h[0] * h[1] + h[3] * h[4]) / d1
        v2 = (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2
        second = _pick_focal(v1, v2, d1, d2)

        # Focal of the first view, from the last column of H.
        d3 = h[0] * h[3] + h[1] * h[4]
        d4 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4]
        v3 = -h[2] * h[5] / d3
        v4 = (h[5] * h[5] - h[2] * h[2]) / d4
        first = _pick_focal(v3, v4, d3, d4)

    return tuple(value for value in (first, second) if value is not None)


def estimate_focal(homographies, width, height):
    """Focal length in pixels for a set of equal-focal views, from their homographies.

    `homographies` are pixel-coordinate homographies between pairs of images of
    the given size -- in the pipeline, the ones on the edges of the spanning
    tree. Every one of them contributes up to two estimates; the median of the
    plausible ones is returned, because a single grazing pair can be far off
    while the bulk of them agrees closely.

    Falls back to FOCAL_FALLBACK_FACTOR * the longest side when nothing valid
    comes out -- a pure translation between two shots, for one, carries no
    focal information at all, and a wrong-but-sane cylinder still stitches
    better than a planar chain across a wide sweep.
    """
    longest = float(max(width, height))
    candidates = []
    for H in homographies:
        if H is None:
            continue
        matrix = np.asarray(H, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            continue
        for focal in focal_candidates(_to_centred(matrix, width, height)):
            if FOCAL_MIN_FACTOR * longest <= focal <= FOCAL_MAX_FACTOR * longest:
                candidates.append(focal)

    if not candidates:
        return FOCAL_FALLBACK_FACTOR * longest
    return float(np.median(candidates))


def _validate_focal(focal):
    """Reject a focal length that could not describe a cylinder."""
    focal = float(focal)
    if not math.isfinite(focal) or focal <= 0.0:
        raise PanoramaError(
            "Could not project onto a cylinder: the focal length is not usable.",
            code="degenerate_homography",
        )
    return focal


def _cylinder_maps(width, height, focal, out_width):
    """Inverse maps taking every cylinder pixel back to its source pixel.

    A pixel (u, v) of the cylinder image stands for the ray at angle
    theta = (u - u0) / f around the cylinder axis and height h = (v - v0) / f
    along it; that ray leaves the camera as (sin theta, h, cos theta) and hits
    the image plane at f * (tan theta, h / cos theta) from the centre. Both
    maps are built by broadcasting, so no Python loop ever touches a pixel.
    """
    theta = (np.arange(out_width, dtype=np.float64) - (out_width - 1) / 2.0) / focal
    rows = np.arange(height, dtype=np.float64) - (height - 1) / 2.0

    map_x = (width - 1) / 2.0 + focal * np.tan(theta)
    map_y = (height - 1) / 2.0 + rows[:, None] / np.cos(theta)[None, :]

    # remap wants two full CV_32FC1 planes, and a broadcast view is not one.
    return (
        np.tile(map_x.astype(np.float32), (height, 1)),
        np.ascontiguousarray(map_y, dtype=np.float32),
    )


def cylindrical_width(width, focal):
    """Width of the cylinder image an image of this width unrolls to.

    The image spans 2 * atan(w / 2f) radians, which is f times that in arc
    length on a cylinder of radius f. The result is forced odd so the cylinder
    has an exact centre column: that column is the ray straight ahead, where
    the projection is the identity.
    """
    half = focal * math.atan(width / (2.0 * focal))
    return 2 * max(1, int(round(half))) + 1


def warp_cylindrical(image, focal):
    """Project one image onto a cylinder of radius `focal`, centred on the image.

    Returns (warped, mask): the cylinder image and its 0/255 validity mask, both
    cropped to the mask's bounding box, so `mask` describes exactly the pixels
    of `warped` that carry content. The image is resampled with INTER_LINEAR and
    the mask with INTER_NEAREST, which keeps it binary; the two therefore agree
    up to the half-pixel fringe every interpolated border has, which is what the
    erosion in blending is there to remove.

    Vertical lines stay vertical and the centre column comes through unchanged;
    everything else is pulled towards the centre, the further out the more, and
    that is what turns the rotation between two shots into a translation.
    """
    focal = _validate_focal(focal)
    height, width = image.shape[:2]

    map_x, map_y = _cylinder_maps(width, height, focal, cylindrical_width(width, focal))
    warped = cv2.remap(
        image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    filled = np.full((height, width), 255, dtype=np.uint8)
    mask = cv2.remap(
        filled, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        raise PanoramaError(
            "The cylindrical projection left nothing of this image.",
            code="degenerate_homography",
        )
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(columns[0]), int(columns[-1]) + 1
    return warped[top:bottom, left:right].copy(), mask[top:bottom, left:right].copy()
