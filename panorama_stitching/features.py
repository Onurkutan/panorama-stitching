"""Stage 1 -- SIFT keypoint detection.

Detection runs on the grayscale version of the image; the rich keypoint
overlay is drawn for the web UI and the CLI outputs.
"""

import cv2

from .errors import PanoramaError


def detect_features(image):
    """Detect SIFT keypoints on a BGR image.

    Returns (keypoints, descriptors, drawn), where `drawn` is the image with
    the rich keypoint overlay. Raises PanoramaError when the image carries no
    usable structure.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) == 0:
        raise PanoramaError(
            "No SIFT features could be found in this image.",
            code="not_enough_features",
        )

    drawn = cv2.drawKeypoints(
        gray,
        keypoints,
        image.copy(),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return keypoints, descriptors, drawn
