"""Stage 5 -- warp, exposure match, feather blend and crop the panorama."""

import cv2
import numpy as np

from .errors import PanoramaError

# A panorama of two overlapping photos stays within a few times the input
# area. A canvas larger than this multiple of the total input pixel count
# means the homography is degenerate, so refuse it before allocating it.
MAX_CANVAS_FACTOR = 8

# Interpolating a warp mixes the black canvas background into the pixels just
# inside the image border, so the outermost ring of a warped image is a dark
# fringe rather than real content. Warped validity masks are eroded by this
# radius to drop it.
MASK_EROSION = 2

# Bounds for the per-channel exposure gain. A real exposure difference between
# two shots of the same scene is small; anything outside this range says the
# overlap statistics are not comparable (moving subject, clipped highlights),
# and scaling by it would do more harm than the mismatch it corrects.
EXPOSURE_GAIN_MIN = 0.8
EXPOSURE_GAIN_MAX = 1.25

# Feather band around the seam. The seam runs along the middle of the overlap
# (equidistant from both image borders) and the two sources are mixed only
# within this half-width of it, proportional to the overlap width and clamped
# to a pixel range. Mixing the whole overlap instead would blend anything that
# moved between the two shots into a semi-transparent ghost.
SEAM_BAND_DIVISOR = 15
SEAM_BAND_MIN = 15
SEAM_BAND_MAX = 60

# Auto-crop: an edge row/column is trimmed only when it is emptier than this
# ratio AND emptier than the box it borders, and never past this fraction of
# the original size. See _crop_valid_area.
CROP_THRESHOLD = 0.8
CROP_MIN_FRACTION = 0.25


def _validate_homography(H):
    """Reject homographies that cannot produce a usable canvas."""
    if H is None:
        raise PanoramaError(
            "No homography available; the images cannot be stitched.",
            code="homography_failed",
        )

    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        raise PanoramaError(
            "Degenerate homography; the images may not overlap enough.",
            code="degenerate_homography",
        )

    # A near-singular matrix collapses the image onto a line or blows it up.
    if abs(np.linalg.det(H)) < 1e-8:
        raise PanoramaError(
            "Degenerate homography; the images may not overlap enough.",
            code="degenerate_homography",
        )
    return H


def _corners(height, width):
    """The four corners of an image, clockwise from the origin."""
    return np.float32([[0, 0], [width, 0], [width, height], [0, height]])


def _canvas_from_points(all_points, input_pixels, pad):
    """Canvas size and translation holding every given point plus a border.

    `all_points` is the (N, 2) stack of the warped corners of every image that
    goes onto the canvas and `input_pixels` their total pixel count, which is
    what the area guard below is measured against.
    """
    # Points behind the camera plane come back as inf/nan; never feed those
    # into the int() conversions below.
    if not np.all(np.isfinite(all_points)):
        raise PanoramaError(
            "Degenerate homography; the images may not overlap enough.",
            code="degenerate_homography",
        )

    xmin, ymin = all_points.min(axis=0)
    xmax, ymax = all_points.max(axis=0)

    # Check the size in float first: a degenerate H can ask for a canvas of
    # billions of pixels, and even computing it as int is pointless then.
    requested_area = float(xmax - xmin + 2 * pad) * float(ymax - ymin + 2 * pad)
    if requested_area > MAX_CANVAS_FACTOR * input_pixels:
        raise PanoramaError(
            "Degenerate homography; the images may not overlap enough.",
            code="degenerate_homography",
        )

    tx = int(np.floor(-xmin)) + pad
    ty = int(np.floor(-ymin)) + pad
    canvas_width = int(np.ceil(xmax - xmin)) + 2 * pad
    canvas_height = int(np.ceil(ymax - ymin)) + 2 * pad

    T = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    return canvas_width, canvas_height, T


def _compute_canvas(H, left_height, left_width, right_height, right_width, pad=2):
    """Compute canvas size and translation from warped left and fixed right image."""
    _validate_homography(H)

    left_corners = _corners(left_height, left_width).reshape(-1, 1, 2)
    left_warped = cv2.perspectiveTransform(left_corners, H).reshape(-1, 2)
    all_points = np.vstack([left_warped, _corners(right_height, right_width)])

    input_pixels = left_height * left_width + right_height * right_width
    return _canvas_from_points(all_points, input_pixels, pad)


def _compute_canvas_set(images, homographies, pad=2):
    """One canvas holding every image of a set warped into the reference plane.

    homographies[k] maps image k into the reference plane, so the reference
    itself carries the identity and its corners are taken as they are -- which
    is also what keeps a two-image set bit-identical to _compute_canvas.
    """
    identity = np.eye(3)
    points = []
    input_pixels = 0
    for image, H in zip(images, homographies, strict=True):
        height, width = image.shape[:2]
        input_pixels += height * width
        corners = _corners(height, width)
        matrix = _validate_homography(H)
        if np.array_equal(matrix, identity):
            points.append(corners)
        else:
            points.append(
                cv2.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
            )
    return _canvas_from_points(np.vstack(points), input_pixels, pad)


def _valid_mask(image):
    """Separate valid image pixels from black background.

    Only a fallback for callers that have no warped mask at hand: a dark but
    real scene pixel is indistinguishable from background this way, and the
    interpolation fringe around a warp counts as valid. Prefer the masks
    produced by _warp_mask / _placement_mask.
    """
    return np.any(image > 0, axis=2).astype(np.uint8) * 255


def _erode_mask(mask, radius=MASK_EROSION):
    """Shave `radius` pixels off a 0/255 mask to drop the interpolation fringe.

    cv2.erode leaves the canvas border untouched, which is what we want here:
    where the warped image runs past the edge of the canvas, the pixels at that
    edge are interior image content and carry no fringe. An image so small that
    the erosion would erase it keeps its mask instead.
    """
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    eroded = cv2.erode(mask, kernel)
    if cv2.countNonZero(eroded) == 0:
        return mask
    return eroded


def _source_mask(height, width, mask):
    """The image's own validity mask, or the full rectangle when it has none."""
    if mask is None:
        return np.full((height, width), 255, dtype=np.uint8)
    return mask


def _warp_mask(height, width, M, canvas_width, canvas_height, mask=None):
    """Validity mask of an image warped by M, as an eroded 0/255 canvas mask.

    The mask is warped with INTER_NEAREST so it stays binary, then eroded: the
    image itself is resampled with a smoothing kernel that pulls the black
    background into the border pixels, and those must not count as content.

    `mask` is the image's own 0/255 validity mask when it has one -- a
    cylindrically projected photo does not fill its rectangle -- and defaults to
    the whole rectangle.
    """
    mask = cv2.warpPerspective(
        _source_mask(height, width, mask),
        M,
        (canvas_width, canvas_height),
        flags=cv2.INTER_NEAREST,
    )
    return _erode_mask(mask)


def _placement_mask(height, width, tx, ty, canvas_width, canvas_height, mask=None):
    """Validity mask of an image copied into the canvas at an integer offset.

    A slice copy resamples nothing, so the full rectangle is exact and needs no
    erosion -- unlike the warped mask above. A mask handed in, on the other
    hand, is the result of a resampling warp of its own and carries the same
    interpolation fringe along its border, so that one is eroded.
    """
    placed = _place_translated(
        _source_mask(height, width, mask), tx, ty, canvas_width, canvas_height
    )
    return placed if mask is None else _erode_mask(placed)


def _bounding_box(mask):
    """Bounding box (top, bottom, left, right) of the non-zero pixels, end-exclusive."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    columns = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(columns[0]), int(columns[-1]) + 1


def _equalize_exposure(warped_left, warped_right, mask_left, mask_right):
    """Scale the warped left image so its overlap exposure matches the right one.

    Two shots of the same scene rarely share an exposure; the gain is a single
    per-channel factor measured on the overlap, applied to the whole left image
    through a lookup table so the corrected part never disagrees with the rest.
    """
    overlap = cv2.bitwise_and(mask_left, mask_right)
    if cv2.countNonZero(overlap) == 0:
        return warped_left

    left_mean = cv2.mean(warped_left, mask=overlap)[:3]
    right_mean = cv2.mean(warped_right, mask=overlap)[:3]

    gains = []
    for left_channel, right_channel in zip(left_mean, right_mean, strict=True):
        # A near-black overlap carries no exposure information; dividing by it
        # would produce a huge gain from rounding noise alone.
        if left_channel < 1.0:
            gains.append(1.0)
            continue
        gains.append(
            float(np.clip(right_channel / left_channel, EXPOSURE_GAIN_MIN, EXPOSURE_GAIN_MAX))
        )

    if all(abs(gain - 1.0) < 1e-3 for gain in gains):
        return warped_left

    scaled = np.arange(256, dtype=np.float32)[None, :, None] * np.float32(gains)
    lut = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
    return cv2.LUT(warped_left, lut)


def _seam_half_width(overlap_width):
    """Half-width in pixels of the mixing band around the seam."""
    return int(np.clip(overlap_width // SEAM_BAND_DIVISOR, SEAM_BAND_MIN, SEAM_BAND_MAX))


def _distance_weights(mask_left, mask_right, box):
    """Feather weights for the left image inside the overlap box.

    d_left and d_right are each pixel's distances to the border of the left and
    right image. Where they are equal lies the seam: the line that stays as
    far as possible from both image borders, so neither outline ever shows up
    as a hard edge and the direction of the ramp follows the geometry (a left
    image that actually landed on the right is handled by the same formula).
    Moving away from the seam, d_left - d_right changes by about two per pixel,
    so the weight reaches 0 or 1 exactly `half_band` pixels out; outside the
    band each side is a single source, which keeps anything that moved between
    the two shots from being blended into a ghost.

    The distance transforms are computed on the whole canvas -- cropping first
    would make the box edges look like image borders and ramp the weights in
    the wrong place -- but only the box is kept, one at a time, so at most one
    full-canvas float array exists at a time.
    """
    top, bottom, left, right = box
    d_left = cv2.distanceTransform(mask_left, cv2.DIST_L2, 3)[top:bottom, left:right].copy()
    d_right = cv2.distanceTransform(mask_right, cv2.DIST_L2, 3)[top:bottom, left:right].copy()

    half_band = _seam_half_width(right - left)
    d_left -= d_right
    d_left /= 4.0 * half_band
    d_left += 0.5
    return np.clip(d_left, 0.0, 1.0, out=d_left)


def _feather_blend(warped_left, warped_right, mask_left=None, mask_right=None):
    """
    Feather-blend two warped BGR images along the seam of their overlap.

    The seam is the line equidistant from both image borders and the sources
    are mixed only in a narrow band around it (see _distance_weights), which
    makes the transition independent of image content and of which side the
    left image actually landed on, keeps the warped outline from showing as a
    hard edge, and leaves moving subjects unblended outside the band.

    Float work is confined to the bounding box of the overlap; everything
    outside it is a uint8 copy.
    """
    if mask_left is None:
        mask_left = _valid_mask(warped_left)
    if mask_right is None:
        mask_right = _valid_mask(warped_right)

    # Exclusive regions first, as a plain uint8 masked copy: the right image
    # overwrites the overlap, which the feathered mix then replaces.
    result = np.zeros_like(warped_left)
    result = cv2.copyTo(warped_left, mask_left, result)
    result = cv2.copyTo(warped_right, mask_right, result)

    overlap = cv2.bitwise_and(mask_left, mask_right)
    box = _bounding_box(overlap)
    if box is None:
        return result

    top, bottom, left, right = box
    weight = _distance_weights(mask_left, mask_right, box)[..., None]

    blended = warped_left[top:bottom, left:right].astype(np.float32)
    blended *= weight
    right_share = warped_right[top:bottom, left:right].astype(np.float32)
    right_share *= 1.0 - weight
    blended += right_share
    del right_share

    np.copyto(
        result[top:bottom, left:right],
        np.clip(blended, 0, 255).astype(np.uint8),
        where=(overlap[top:bottom, left:right] > 0)[..., None],
    )
    return result


def _region_sum(integral, top, bottom, left, right):
    """Valid pixel count inside the inclusive box, read off the integral image."""
    return int(
        integral[bottom + 1, right + 1]
        - integral[top, right + 1]
        - integral[bottom + 1, left]
        + integral[top, left]
    )


def _row_ratio(integral, row, left, right):
    """Occupancy ratio of one row over the column range [left, right]."""
    return _region_sum(integral, row, row, left, right) / (right - left + 1)


def _column_ratio(integral, column, top, bottom):
    """Occupancy ratio of one column over the row range [top, bottom]."""
    return _region_sum(integral, top, bottom, column, column) / (bottom - top + 1)


def _box_ratio(integral, top, bottom, left, right):
    """Occupancy ratio of the whole crop box."""
    return _region_sum(integral, top, bottom, left, right) / (
        (bottom - top + 1) * (right - left + 1)
    )


def _crop_valid_area(panorama, mask=None):
    """
    Trim the black border a warp leaves around the panorama.

    An edge row or column is dropped only when it is emptier than CROP_THRESHOLD
    *and* emptier than the box it borders. The second condition is what keeps
    the rule stable: a hole pattern that runs through the whole image (every
    third row black, one black half) is no worse at the edge than in the
    middle, so trimming it away would not improve anything and does not start.
    A hard floor of CROP_MIN_FRACTION of each side caps the damage of any
    remaining pathological case, and the passes are bounded at two. On the
    bundled examples this lands on the same crop box as the older unbounded
    loop; what it changes is what happens on masks that loop collapsed on.

    All occupancy ratios are read off a 2-D prefix sum (integral image) built
    once, so every query costs O(1) regardless of the region size. Pass the
    union of the two validity masks as `mask`; without it the mask is guessed
    from the pixels, and dark scene content counts as border.
    """
    if mask is None:
        mask = _valid_mask(panorama)
    if not np.any(mask):
        return panorama

    occupancy = (mask > 0).astype(np.uint8)
    # integral[y, x] = number of valid pixels in occupancy[:y, :x] (zero padded).
    integral = cv2.integral(occupancy, sdepth=cv2.CV_32S)

    height, width = occupancy.shape
    top, bottom = 0, height - 1
    left, right = 0, width - 1
    min_height = max(1, int(height * CROP_MIN_FRACTION))
    min_width = max(1, int(width * CROP_MIN_FRACTION))

    def is_empty(ratio, box_ratio):
        return ratio < CROP_THRESHOLD and ratio < box_ratio

    for _ in range(2):
        changed = False

        while bottom - top + 1 > min_height:
            box = _box_ratio(integral, top, bottom, left, right)
            if is_empty(_row_ratio(integral, top, left, right), box):
                top += 1
            elif is_empty(_row_ratio(integral, bottom, left, right), box):
                bottom -= 1
            else:
                break
            changed = True

        while right - left + 1 > min_width:
            box = _box_ratio(integral, top, bottom, left, right)
            if is_empty(_column_ratio(integral, left, top, bottom), box):
                left += 1
            elif is_empty(_column_ratio(integral, right, top, bottom), box):
                right -= 1
            else:
                break
            changed = True

        if not changed:
            break

    return panorama[top : bottom + 1, left : right + 1]


def _integer_translation(T):
    """Return (tx, ty) when T is a pure integer pixel translation, else None."""
    if T is None or np.asarray(T).shape != (3, 3):
        return None
    T = np.asarray(T, dtype=np.float64)
    tx, ty = T[0, 2], T[1, 2]
    expected = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
    if not np.array_equal(T, expected):
        return None
    if tx != int(tx) or ty != int(ty):
        return None
    return int(tx), int(ty)


def _place_translated(image, tx, ty, canvas_width, canvas_height):
    """Copy `image` into a zeroed canvas at integer offset (tx, ty)."""
    canvas = np.zeros((canvas_height, canvas_width) + image.shape[2:], dtype=image.dtype)
    h, w = image.shape[:2]

    # Clip to the canvas so an offset that pushes the image out never throws.
    y0, y1 = max(0, ty), min(canvas_height, ty + h)
    x0, x1 = max(0, tx), min(canvas_width, tx + w)
    if y1 > y0 and x1 > x0:
        canvas[y0:y1, x0:x1] = image[y0 - ty : y1 - ty, x0 - tx : x1 - tx]
    return canvas


def stitch_images(left_bgr, right_bgr, H, match_exposure=True):
    """
    Stage 5 -- panorama stitch:
    1. Warp the left image into the right image plane with H.
    2. Align both images on a shared canvas.
    3. Match the exposure of the two shots on their overlap.
    4. Feather-blend the overlap.
    5. Auto-crop the resulting black borders.

    H maps left-image coordinates into the right-image plane. Pass
    match_exposure=False to keep the left image's own exposure.

    This is the two-image case of stitch_set_images: the right image is the
    reference (identity homography, placed by a slice copy) and the left one
    is the single image composited onto it. Both paths therefore produce the
    very same panorama, byte for byte.
    """
    return stitch_set_images(
        [left_bgr, right_bgr],
        [H, np.eye(3)],
        reference_index=1,
        order=[0],
        match_exposure=match_exposure,
    )


def stitch_set_images(
    images, homographies, reference_index, order, match_exposure=True, masks=None
):
    """
    Stage 5 for a whole set -- warp every photo into the reference plane and
    composite them onto one canvas, outward from the reference.

    homographies[k] maps image k into the reference plane; the reference's own
    entry is the identity. `order` is the sequence in which the other images
    join the composite -- breadth-first from the reference, so each of them is
    blended against a composite it actually overlaps. The reference index may
    appear in `order`; it is skipped there.

    The reference is placed with an integer slice copy, so its validity mask is
    an exact rectangle; every other image is warped with INTER_CUBIC and gets
    an eroded mask that leaves out the interpolation fringe. The growing
    composite is the fixed side of every step: each new image is scaled towards
    the composite's exposure and feathered into it, and earlier images are
    never resampled or re-graded again. The union of all validity masks drives
    the final auto-crop.

    `masks` is an optional per-image 0/255 validity mask, for images that do not
    fill their own rectangle -- what the cylindrical projection produces. Each
    image's own mask is then warped (or placed) instead of the full rectangle,
    so the black corners of a cylinder image never count as content. Without it
    every image is taken to be fully valid, which is the two-image case and the
    planar set.
    """
    canvas_width, canvas_height, T = _compute_canvas_set(images, homographies)

    reference = images[reference_index]
    reference_height, reference_width = reference.shape[:2]
    reference_mask = None if masks is None else masks[reference_index]

    # T is a whole-pixel translation by construction, so the reference can be
    # copied in rather than resampled onto itself. Fall back to a warp if it
    # is ever not integral.
    translation = _integer_translation(T)
    if translation is None:
        composite = cv2.warpPerspective(
            reference, T, (canvas_width, canvas_height), flags=cv2.INTER_CUBIC
        )
        composite_mask = _warp_mask(
            reference_height, reference_width, T, canvas_width, canvas_height, reference_mask
        )
    else:
        tx, ty = translation
        composite = _place_translated(reference, tx, ty, canvas_width, canvas_height)
        composite_mask = _placement_mask(
            reference_height, reference_width, tx, ty, canvas_width, canvas_height, reference_mask
        )

    for index in order:
        if index == reference_index:
            continue
        image = images[index]
        height, width = image.shape[:2]
        M = T @ homographies[index]
        warped = cv2.warpPerspective(image, M, (canvas_width, canvas_height), flags=cv2.INTER_CUBIC)
        mask = _warp_mask(
            height, width, M, canvas_width, canvas_height, None if masks is None else masks[index]
        )

        if match_exposure:
            warped = _equalize_exposure(warped, composite, mask, composite_mask)
        composite = _feather_blend(warped, composite, mask, composite_mask)
        composite_mask = cv2.bitwise_or(mask, composite_mask)

    return _crop_valid_area(composite, composite_mask)
