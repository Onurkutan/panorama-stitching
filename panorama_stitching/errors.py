"""Shared error type for the panorama stitching pipeline.

Every library module (features, matching, homography, blending, pipeline)
raises PanoramaError instead of printing and returning None, so callers -- the
web app and the CLI -- can decide how to report the failure. Each error also
carries a short, stable `code` that clients can branch on without parsing the
human-readable message.
"""

# Stable identifiers attached to a PanoramaError. The web front end maps them
# onto localized texts, so add to this list rather than changing the strings.
ERROR_CODES = (
    "image_unreadable",
    "not_enough_features",
    "not_enough_matches",
    "homography_failed",
    "degenerate_homography",
    "output_write_failed",
    "example_not_found",
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
    """

    def __init__(self, message, code="unexpected"):
        super().__init__(message)
        self.code = code
