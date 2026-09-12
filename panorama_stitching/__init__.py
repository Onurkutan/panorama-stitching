"""Panorama stitching: SIFT features, FLANN matching, RANSAC homography, blending.

The stages live in dedicated modules (features, matching, homography, blending,
and projection for the cylinder a wide sweep is stitched on); `pipeline.stitch_pair`
runs all of them on one image pair and `pipeline.stitch_set` on 2..MAX_IMAGES photos
given in any order. The names below are re-exported for the common case:

    from panorama_stitching import PanoramaError, stitch_pair, stitch_set
"""

from .errors import PanoramaError
from .pipeline import (
    EXAMPLES,
    MAX_IMAGES,
    PROGRESS_STAGES,
    read_image,
    stitch_pair,
    stitch_set,
)

__version__ = "1.0.0"

__all__ = [
    "EXAMPLES",
    "MAX_IMAGES",
    "PROGRESS_STAGES",
    "PanoramaError",
    "__version__",
    "read_image",
    "stitch_pair",
    "stitch_set",
]
