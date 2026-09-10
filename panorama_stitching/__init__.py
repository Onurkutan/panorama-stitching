"""Panorama stitching: SIFT features, FLANN matching, RANSAC homography, blending.

The stages live in dedicated modules (features, matching, homography, blending)
and `pipeline.stitch_pair` runs all of them on one image pair. The names below
are re-exported for the common case:

    from panorama_stitching import PanoramaError, stitch_pair
"""

from .errors import PanoramaError
from .pipeline import EXAMPLES, stitch_pair

__version__ = "1.0.0"

__all__ = ["EXAMPLES", "PanoramaError", "__version__", "stitch_pair"]
