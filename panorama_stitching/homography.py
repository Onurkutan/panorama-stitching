"""Stage 3 -- RANSAC homography estimation between the two views."""

import logging

import cv2
import numpy as np

from .errors import PanoramaError

# Minimum number of ratio-test matches required for a reliable homography.
# Shared with the pipeline so both stages use the very same gate.
MIN_MATCH_COUNT = 10

_logger = logging.getLogger(__name__)


def estimate_homography(
    kp1,
    kp2,
    good_matches,
    left_image,
    right_image,
    inlier_output_path=None,
):
    """Estimate the left -> right homography with RANSAC.

    Returns (H, mask). Raises PanoramaError when there are too few matches or
    when RANSAC cannot fit a model. The RANSAC inlier visualization is only
    drawn and saved when inlier_output_path is given (nothing is written to
    the current working directory by default).
    """
    # 1. Safety gate: require at least MIN_MATCH_COUNT matches for a reliable result
    if len(good_matches) < MIN_MATCH_COUNT:
        raise PanoramaError(
            f"Not enough matches. Required: {MIN_MATCH_COUNT}, found: {len(good_matches)}",
            code="not_enough_matches",
        )

    # Collect point coordinates
    left_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    right_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute the homography (warp matrix) with RANSAC.
    # RANSAC draws random minimal sample sets, so without a fixed seed the
    # inlier mask -- and every image derived from it -- changes from run to
    # run. Seeding right before the call keeps outputs repeatable for tests.
    cv2.setRNGSeed(0)
    H, mask = cv2.findHomography(left_points, right_points, cv2.RANSAC, 5.0)

    if H is None or mask is None:
        raise PanoramaError(
            "Could not compute the homography matrix.",
            code="homography_failed",
        )

    inlier_count = int(np.sum(mask))

    # 2. Visualization: draw only matches that passed RANSAC.
    # mask holds values like [1, 0, 1, 1...]; 1 = inlier, 0 = outlier
    if inlier_output_path:
        matches_mask = mask.ravel().tolist()

        # Draw inliers (mask == 1) in green
        draw_params = dict(
            matchColor=(0, 255, 0),  # Inlier match lines in green
            singlePointColor=None,
            matchesMask=matches_mask,  # Draw only RANSAC inliers
            flags=2,
        )

        # Draw the filtered matches on the image
        inlier_preview = cv2.drawMatches(
            left_image, kp1, right_image, kp2, good_matches, None, **draw_params
        )

        # Save the filtered match visualization
        cv2.imwrite(inlier_output_path, inlier_preview)
        _logger.info(
            "RANSAC kept %d of %d matches as inliers. Visualization saved: %s",
            inlier_count,
            len(good_matches),
            inlier_output_path,
        )
    else:
        _logger.info(
            "RANSAC kept %d of %d matches as inliers.",
            inlier_count,
            len(good_matches),
        )

    return H, mask
