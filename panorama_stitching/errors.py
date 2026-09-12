"""Shared error type for the panorama stitching pipeline.

Every library module (features, matching, homography, blending, pipeline)
raises PanoramaError instead of printing and returning None, so callers -- the
web app and the CLI -- can decide how to report the failure. Each error also
carries a short, stable `code` that clients can branch on without parsing the
human-readable message, and optionally a small `details` dictionary naming the
image or the pair the failure belongs to.
"""

# Stable identifiers attached to a PanoramaError. The web front end maps them
# onto localized texts, so add to this list rather than changing the strings.
ERROR_CODES = (
    "image_unreadable",
    "heic_unsupported",
    "not_enough_features",
    "not_enough_matches",
    "homography_failed",
    "degenerate_homography",
    "output_write_failed",
    "example_not_found",
    "too_few_images",
    "too_many_images",
    "image_not_connected",
    "form_invalid",
    "upload_missing",
    "upload_not_image",
    "upload_too_large",
    "unexpected",
)


class PanoramaError(RuntimeError):
    """A stitching stage could not produce a usable result.

    `code` is one of ERROR_CODES; it defaults to "unexpected" so an error
    raised without one still has the attribute.

    `details` is an optional JSON-serialisable dictionary that points at the
    input the failure belongs to: {"image": k} for a single photo and
    {"pair": [i, j]} for a photo pair, both 1-based over the input order. It
    is None when the failure is not tied to a particular input.
    """

    def __init__(self, message, code="unexpected", details=None):
        super().__init__(message)
        self.code = code
        self.details = details
