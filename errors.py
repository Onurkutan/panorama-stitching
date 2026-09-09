"""Shared error type for the panorama stitching pipeline.

Library modules (matcher, homografi, birlestirme, panorama_pipeline) raise
PanoramaError instead of printing and returning None, so callers -- the web
app and the CLI -- can decide how to report the failure. The messages are
Turkish because they are shown to the user in the web UI.
"""


class PanoramaError(RuntimeError):
    """A stitching stage could not produce a usable result."""
