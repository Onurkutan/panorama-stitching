"""Stage 2 -- descriptor matching with FLANN and Lowe's ratio test."""

import logging

import cv2

from .errors import PanoramaError

# Lowe's ratio test threshold: a match is kept only when the best distance is
# clearly smaller than the second best one.
RATIO_THRESHOLD = 0.7

_logger = logging.getLogger(__name__)


def match_features(kp1, des1, kp2, des2, ratio_threshold=RATIO_THRESHOLD):
    """Match two SIFT descriptor sets with FLANN + Lowe's ratio test.

    Raises PanoramaError when the descriptors are missing or too few for a
    k=2 nearest neighbour search (flat or tiny images), instead of letting
    OpenCV throw a raw cv2.error.
    """
    # 0. Safety gate: knnMatch(k=2) needs at least two descriptors on both sides
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        raise PanoramaError(
            "Not enough SIFT descriptors; the image may be too flat or too small.",
            code="not_enough_features",
        )

    # 1. FLANN parameters (standard settings for SIFT)
    # FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)

    # 2. Create the matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 3. k-nearest neighbour matching (k=2)
    # For each point, find the two best matches.
    # The FLANN kd-tree index is randomized, so the approximate neighbours
    # differ slightly between runs. Seed the global RNG first to keep the
    # match set -- and therefore every downstream output -- repeatable.
    cv2.setRNGSeed(0)
    matches = flann.knnMatch(des1, des2, k=2)

    # 4. Lowe's ratio test (quality filter)
    # Keep a match only if the best distance is clearly smaller than the second best.
    good_matches = []
    for pair in matches:
        # Near the edges of the index FLANN may return fewer than two
        # neighbours; such pairs cannot be ratio tested, so skip them.
        if len(pair) != 2:
            continue
        best, second_best = pair
        if best.distance < ratio_threshold * second_best.distance:
            good_matches.append(best)

    _logger.info("Kept %d good matches out of %d.", len(good_matches), len(matches))
    return good_matches
