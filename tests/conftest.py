"""Shared fixtures for the test suite."""

import uuid

import cv2
import numpy as np
import pytest

import panorama_pipeline

PROJECT_DIR = panorama_pipeline.PROJECT_DIR


def _textured_image(width, height, seed=0):
    """A synthetic BGR image with blob-like texture SIFT can key on.

    Plain noise gives SIFT nothing stable to lock onto, and a flat gradient
    is not distinctive either. Random blurred blobs of random size and
    colour give plenty of corner- and blob-like structure without needing a
    real photograph.
    """
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 40, dtype=np.uint8)
    for _ in range(250):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        r = int(rng.integers(4, 22))
        color = tuple(int(c) for c in rng.integers(40, 256, size=3))
        cv2.circle(img, (x, y), r, color, -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


@pytest.fixture(scope="session")
def overlapping_pair():
    """Two crops of one textured scene sharing a known pixel-column overlap.

    left  = scene[:, 0:left_w]
    right = scene[:, shift:shift + right_w]

    A point at column x in `left` (x >= shift) is the same scene point as
    column (x - shift) in `right` -- a pure horizontal-translation
    relationship, exposed here as `shift`.
    """
    height, scene_w = 260, 420
    scene = _textured_image(scene_w, height, seed=1)
    left_w, right_w = 260, 260
    shift = scene_w - right_w  # last `right_w` columns of the scene
    left = scene[:, 0:left_w].copy()
    right = scene[:, shift : shift + right_w].copy()
    return {"left": left, "right": right, "shift": shift}


@pytest.fixture(scope="session")
def sift_descriptors(overlapping_pair):
    """Real SIFT keypoints/descriptors detected on the synthetic left crop."""
    gray = cv2.cvtColor(overlapping_pair["left"], cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    assert descriptors is not None and len(descriptors) >= 20
    return keypoints, descriptors


@pytest.fixture(scope="session")
def clock_pair_paths():
    """Paths to the bundled Clock example pair (loaded via stitch_pair's max_side)."""
    left = PROJECT_DIR / "images" / "Clock" / "sol1.jpg"
    right = PROJECT_DIR / "images" / "Clock" / "sag1.jpg"
    assert left.exists() and right.exists()
    return left, right


def _encode_multipart(fields, files):
    """Build a multipart/form-data body by hand.

    fields: {name: value}
    files: {name: (filename, content_bytes)}
    Returns (body_bytes, content_type_header_value).
    """
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
            ).encode()
        )
    for name, (filename, content) in files.items():
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        parts.append(header + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


@pytest.fixture
def multipart_body():
    """Factory fixture: multipart_body(fields, files) -> (body_bytes, content_type)."""
    return _encode_multipart
