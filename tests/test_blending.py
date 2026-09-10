import tracemalloc

import cv2
import numpy as np
import pytest

from panorama_stitching import blending
from panorama_stitching.errors import PanoramaError


def test_compute_canvas_identity():
    left_h, left_w = 100, 150
    right_h, right_w = 120, 90
    pad = 2

    width, height, T = blending._compute_canvas(
        np.eye(3), left_h, left_w, right_h, right_w, pad=pad
    )

    assert width == max(left_w, right_w) + 2 * pad
    assert height == max(left_h, right_h) + 2 * pad
    tx, ty = T[0, 2], T[1, 2]
    assert tx == pad and ty == pad
    assert np.array_equal(T, np.float32([[1, 0, pad], [0, 1, pad], [0, 0, 1]]))


def test_compute_canvas_extreme_scale_raises():
    # Not singular (det = 2500), but the resulting canvas is far larger than
    # 8x the input pixel count, so the area guard must reject it.
    H = np.diag([50.0, 50.0, 1.0])
    with pytest.raises(PanoramaError):
        blending._compute_canvas(H, 100, 100, 100, 100)


def test_compute_canvas_nan_raises():
    H = np.eye(3)
    H[0, 0] = np.nan
    with pytest.raises(PanoramaError):
        blending._compute_canvas(H, 100, 100, 100, 100)


def test_compute_canvas_none_raises():
    with pytest.raises(PanoramaError):
        blending._compute_canvas(None, 100, 100, 100, 100)


def test_crop_valid_area_all_black_returns_same_object():
    img = np.zeros((30, 40, 3), dtype=np.uint8)
    result = blending._crop_valid_area(img)
    assert result is img


def test_crop_valid_area_l_shaped_border():
    h, w = 50, 60
    r0, c0 = 8, 10  # top and left border kept under the 20% occupancy guard
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[r0:, c0:] = 150

    result = blending._crop_valid_area(img)

    assert result.shape[:2] == (h - r0, w - c0)
    assert np.array_equal(result, img[r0:, c0:])


def test_crop_valid_area_fully_valid_unchanged():
    rng = np.random.default_rng(3)
    img = rng.integers(1, 256, size=(20, 25, 3), dtype=np.uint8)
    result = blending._crop_valid_area(img)
    assert result.shape == img.shape
    assert np.array_equal(result, img)


def test_feather_blend():
    h, w = 50, 200
    left_color = np.array([200, 50, 50], dtype=np.uint8)
    right_color = np.array([50, 50, 200], dtype=np.uint8)

    warped_left = np.zeros((h, w, 3), dtype=np.uint8)
    warped_left[:, 0:120] = left_color

    warped_right = np.zeros((h, w, 3), dtype=np.uint8)
    warped_right[:, 80:200] = right_color

    result = blending._feather_blend(warped_left, warped_right)

    assert result.dtype == np.uint8
    assert np.array_equal(result[:, 0:80], np.broadcast_to(left_color, (h, 80, 3)))
    assert np.array_equal(result[:, 120:200], np.broadcast_to(right_color, (h, 80, 3)))

    centre_col = 99  # ~ midpoint of the overlap columns [80, 119]
    expected_mean = (left_color.astype(np.float32) + right_color.astype(np.float32)) / 2
    assert np.allclose(result[:, centre_col].astype(np.float32), expected_mean, atol=5)


def test_place_translated_matches_warp_perspective():
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
    tx, ty = 15, 8
    canvas_w, canvas_h = 80, 60

    result = blending._place_translated(img, tx, ty, canvas_w, canvas_h)

    T = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    expected = cv2.warpPerspective(img, T, (canvas_w, canvas_h))

    assert np.array_equal(result, expected)


# --------------------------------------------------------------- validity masks


def _skewed_homography():
    """Rotation plus a little perspective: the warped border then runs
    diagonally across the canvas and interpolation has a black background to
    mix into the border pixels."""
    return np.float32([[0.92, -0.18, 120.0], [0.10, 0.95, 20.0], [1e-5, 2e-5, 1.0]])


def test_warp_mask_leaves_out_dark_fringe():
    flat = np.full((300, 400, 3), 200, dtype=np.uint8)
    H = _skewed_homography()
    width, height, T = blending._compute_canvas(H, 300, 400, 300, 400)
    M = T @ H
    warped = cv2.warpPerspective(flat, M, (width, height), flags=cv2.INTER_CUBIC)

    mask = blending._warp_mask(300, 400, M, width, height)
    valid = mask > 0
    assert np.count_nonzero(valid) > 90000
    # Every pixel the mask calls valid is real content, not a dark fringe.
    assert warped[valid].min() >= 190

    # The plain "any channel > 0" rule does let the fringe through, which is
    # exactly why the warped mask exists.
    coarse = blending._valid_mask(warped) > 0
    assert warped[coarse].min() < 60
    assert np.count_nonzero(coarse) > np.count_nonzero(valid)


def test_warp_mask_erodes_the_border():
    radius = blending.MASK_EROSION
    # A 40x60 image translated to (5, 4) on a 60x80 canvas: the whole border of
    # the placed rectangle is interior to the canvas, so all of it is eroded.
    M = np.float32([[1, 0, 5], [0, 1, 4], [0, 0, 1]])
    mask = blending._warp_mask(40, 60, M, 80, 60)

    expected = np.zeros((60, 80), dtype=np.uint8)
    expected[4 + radius : 44 - radius, 5 + radius : 65 - radius] = 255
    assert np.array_equal(mask, expected)


def test_warp_mask_does_not_erase_a_tiny_image():
    # Eroding a 3x3 image by 2 px would leave nothing; the mask must survive.
    assert np.count_nonzero(blending._warp_mask(3, 3, np.eye(3), 3, 3)) > 0


def test_placement_mask_is_an_exact_rectangle():
    mask = blending._placement_mask(20, 30, 5, 4, 60, 40)

    assert np.count_nonzero(mask) == 20 * 30
    assert np.array_equal(mask[4:24, 5:35], np.full((20, 30), 255, dtype=np.uint8))


def test_feather_blend_with_given_mask_keeps_black_content():
    h, w = 50, 200
    left_color = np.array([180, 180, 180], dtype=np.uint8)
    right_color = np.array([200, 50, 50], dtype=np.uint8)

    warped_left = np.zeros((h, w, 3), dtype=np.uint8)
    warped_left[:, 0:120] = left_color
    warped_left[10:20, 95:105] = 0  # genuinely black scene content, inside the overlap
    warped_right = np.zeros((h, w, 3), dtype=np.uint8)
    warped_right[:, 80:200] = right_color

    mask_left = np.zeros((h, w), dtype=np.uint8)
    mask_left[:, 0:120] = 255
    mask_right = np.zeros((h, w), dtype=np.uint8)
    mask_right[:, 80:200] = 255

    with_mask = blending._feather_blend(warped_left, warped_right, mask_left, mask_right)
    without_mask = blending._feather_blend(warped_left, warped_right)

    # With a real mask the black patch is blended like any other pixel and
    # darkens the result; the "> 0" fallback drops it and keeps the right image.
    assert with_mask[10:20, 95:105].mean() < 0.75 * float(right_color.mean())
    assert np.array_equal(without_mask[10:20, 95:105], np.broadcast_to(right_color, (10, 10, 3)))


# ------------------------------------------------------------- feather weights


def test_feather_blend_flips_direction_when_left_lands_on_the_right():
    h, w = 50, 200
    left_color = np.array([50, 50, 200], dtype=np.uint8)
    right_color = np.array([200, 50, 50], dtype=np.uint8)

    # The image passed as "left" landed on the RIGHT half of the canvas.
    warped_left = np.zeros((h, w, 3), dtype=np.uint8)
    warped_left[:, 80:200] = left_color
    warped_right = np.zeros((h, w, 3), dtype=np.uint8)
    warped_right[:, 0:120] = right_color

    result = blending._feather_blend(warped_left, warped_right)

    assert np.array_equal(result[:, 0:80], np.broadcast_to(right_color, (h, 80, 3)))
    assert np.array_equal(result[:, 120:200], np.broadcast_to(left_color, (h, 80, 3)))
    # Each source fades out where it ends, whichever side it landed on: at the
    # first overlap column the result still belongs to the right image.
    assert np.allclose(result[:, 80].astype(np.float32), right_color, atol=12)
    assert np.allclose(result[:, 119].astype(np.float32), left_color, atol=12)
    expected = (left_color.astype(np.float32) + right_color.astype(np.float32)) / 2
    assert np.allclose(result[:, 99].astype(np.float32), expected, atol=8)


def test_feather_blend_approaches_its_source_at_the_overlap_edges():
    h, w = 50, 200
    left_color = np.array([200, 50, 50], dtype=np.uint8)
    right_color = np.array([50, 50, 200], dtype=np.uint8)
    warped_left = np.zeros((h, w, 3), dtype=np.uint8)
    warped_left[:, 0:120] = left_color
    warped_right = np.zeros((h, w, 3), dtype=np.uint8)
    warped_right[:, 80:200] = right_color

    result = blending._feather_blend(warped_left, warped_right)

    # No step at either end of the overlap: that step is the visible seam.
    assert np.allclose(result[:, 80].astype(np.float32), left_color, atol=12)
    assert np.allclose(result[:, 119].astype(np.float32), right_color, atol=12)


# ------------------------------------------------------- exposure compensation


def _ramp(height, width, base=120):
    """A smooth mid-grey ramp: bright enough to scale without clipping."""
    x = np.linspace(base - 20, base + 20, width, dtype=np.float32)
    return np.repeat(np.repeat(x[None, :, None], height, axis=0), 3, axis=2).astype(np.uint8)


def test_equalize_exposure_aligns_the_overlap_means():
    warped_left = _ramp(40, 60)
    warped_right = np.clip(warped_left.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
    mask = np.full((40, 60), 255, dtype=np.uint8)

    corrected = blending._equalize_exposure(warped_left, warped_right, mask, mask)

    assert abs(cv2.mean(corrected, mask=mask)[0] - cv2.mean(warped_right, mask=mask)[0]) < 1.0
    assert corrected.mean() > warped_left.mean()


def test_equalize_exposure_clamps_the_gain():
    warped_left = np.full((20, 20, 3), 40, dtype=np.uint8)
    warped_right = np.full((20, 20, 3), 200, dtype=np.uint8)  # ratio 5.0
    mask = np.full((20, 20), 255, dtype=np.uint8)

    corrected = blending._equalize_exposure(warped_left, warped_right, mask, mask)

    assert np.all(corrected == round(40 * blending.EXPOSURE_GAIN_MAX))


def test_equalize_exposure_leaves_a_disjoint_pair_alone():
    warped_left = np.full((20, 40, 3), 100, dtype=np.uint8)
    warped_right = np.full((20, 40, 3), 200, dtype=np.uint8)
    mask_left = np.zeros((20, 40), dtype=np.uint8)
    mask_left[:, :15] = 255
    mask_right = np.zeros((20, 40), dtype=np.uint8)
    mask_right[:, 25:] = 255

    result = blending._equalize_exposure(warped_left, warped_right, mask_left, mask_right)
    assert result is warped_left


def test_stitch_images_matches_the_exposure():
    rng = np.random.default_rng(11)
    scene = np.full((120, 300, 3), 60, dtype=np.uint8)
    for _ in range(120):
        cv2.circle(
            scene,
            (int(rng.integers(0, 300)), int(rng.integers(0, 120))),
            int(rng.integers(4, 18)),
            tuple(int(c) for c in rng.integers(60, 200, size=3)),
            -1,
        )
    shift = 120
    left = scene[:, 0:200].copy()
    right = np.clip(scene[:, shift:300].astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
    H = np.float32([[1, 0, -shift], [0, 1, 0], [0, 0, 1]])

    enabled = blending.stitch_images(left, right, H)
    disabled = blending.stitch_images(left, right, H, match_exposure=False)

    assert enabled.shape == disabled.shape
    assert enabled.mean() > disabled.mean() * 1.03


# ------------------------------------------------------------------- auto-crop


def test_crop_valid_area_does_not_collapse_a_half_black_frame():
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    img[:, :30] = 200

    result = blending._crop_valid_area(img)

    # The empty half goes, the full half is kept whole.
    assert result.shape[:2] == (40, 30)
    assert np.array_equal(result, img[:, :30])


def test_crop_valid_area_does_not_collapse_on_sparse_rows():
    img = np.full((40, 60, 3), 200, dtype=np.uint8)
    img[::3] = 0

    result = blending._crop_valid_area(img)

    # Only the empty first and last rows go: a hole pattern that runs through
    # the whole image is no worse at the edge than in the middle, so trimming
    # the edge would not improve the box.
    assert result.shape[:2] == (38, 60)


def test_crop_valid_area_with_mask_keeps_dark_content():
    img = np.full((40, 60, 3), 30, dtype=np.uint8)
    img[:, :10] = 0  # black, but real scene content
    mask = np.full((40, 60), 255, dtype=np.uint8)

    assert blending._crop_valid_area(img, mask).shape == img.shape
    # Without the mask the black block is read as border and trimmed away.
    assert blending._crop_valid_area(img).shape[:2] == (40, 50)


def test_crop_valid_area_respects_the_size_floor():
    # Pathological input: occupancy grows column by column, so every edge is
    # emptier than the box behind it and a purely greedy rule keeps eating.
    occupancy = np.zeros((40, 60), dtype=np.uint8)
    for column in range(60):
        occupancy[: int(40 * (column + 1) / 60), column] = 255
    img = np.dstack([occupancy] * 3)

    result = blending._crop_valid_area(img, occupancy)

    assert result.shape[0] >= 40 * blending.CROP_MIN_FRACTION
    assert result.shape[1] >= 60 * blending.CROP_MIN_FRACTION


# ---------------------------------------------------------------------- memory


def test_stitch_images_allocates_no_canvas_sized_floats():
    rng = np.random.default_rng(13)
    left = rng.integers(0, 256, size=(400, 500, 3), dtype=np.uint8)
    right = rng.integers(0, 256, size=(400, 500, 3), dtype=np.uint8)
    H = np.float32([[1, 0, -380], [0, 1, 0], [0, 0, 1]])

    width, height, _ = blending._compute_canvas(H, 400, 500, 400, 500)
    canvas_bytes = width * height * 3

    blending.stitch_images(left, right, H)  # warm the allocator caches up
    tracemalloc.start()
    tracemalloc.reset_peak()
    blending.stitch_images(left, right, H)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Blending the whole canvas in float32 costs ~12 bytes per pixel per
    # temporary; keeping the float work inside the overlap box holds the peak
    # near a handful of uint8 canvases. (numpy reports its allocations to
    # tracemalloc, so this measures the arrays that matter here.)
    assert peak < 8 * canvas_bytes


def test_distance_weights_single_source_outside_the_seam_band():
    # Two 200-px wide rectangles overlapping by 100 px: the seam is the
    # overlap's middle column, and beyond the band each side is pure.
    height, width = 60, 300
    mask_left = np.zeros((height, width), np.uint8)
    mask_right = np.zeros((height, width), np.uint8)
    mask_left[:, :200] = 255
    mask_right[:, 100:] = 255
    box = (0, height, 100, 200)

    weight = blending._distance_weights(mask_left, mask_right, box)
    half_band = blending._seam_half_width(100)

    assert weight.shape == (height, 100)
    middle_row = weight[height // 2]
    assert abs(middle_row[50] - 0.5) < 0.05
    assert np.all(middle_row[: 50 - half_band - 1] == 1.0)
    assert np.all(middle_row[50 + half_band + 1 :] == 0.0)
    assert np.all(np.diff(middle_row) <= 0)
