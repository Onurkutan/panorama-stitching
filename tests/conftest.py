"""Shared fixtures for the test suite."""

import math
import uuid

import cv2
import numpy as np
import pytest

from panorama_stitching import pipeline

PROJECT_DIR = pipeline.PROJECT_DIR


def _textured_image(width, height, seed=0, blobs=250):
    """A synthetic BGR image with blob-like texture SIFT can key on.

    Plain noise gives SIFT nothing stable to lock onto, and a flat gradient
    is not distinctive either. Random blurred blobs of random size and
    colour give plenty of corner- and blob-like structure without needing a
    real photograph. `blobs` scales the density with the image area.
    """
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 40, dtype=np.uint8)
    for _ in range(blobs):
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


# Crops of one synthetic scene: 420 px wide, 240 px apart, so consecutive
# crops share 180 px of it. Written to disk once per session -- the set
# pipeline takes paths, not arrays.
CROP_WIDTH = 420
CROP_STRIDE = 240


@pytest.fixture(scope="session")
def crop_set_paths(tmp_path_factory):
    """Three overlapping crops of one scene, in their true left-to-right order."""
    scene = _textured_image(900, 300, seed=1, blobs=620)
    directory = tmp_path_factory.mktemp("crop_set")
    paths = []
    for index in range(3):
        start = index * CROP_STRIDE
        path = directory / f"crop_{index + 1}.jpg"
        cv2.imwrite(str(path), scene[:, start : start + CROP_WIDTH].copy())
        paths.append(path)
    return paths


@pytest.fixture(scope="session")
def unrelated_image_path(tmp_path_factory):
    """A photo of a different scene: plenty of features, none of them shared."""
    path = tmp_path_factory.mktemp("unrelated") / "unrelated.jpg"
    cv2.imwrite(str(path), _textured_image(CROP_WIDTH, 300, seed=99, blobs=620))
    return path


# A camera that only turns: four frames of one scene, 30 degrees apart, all
# taken with the same focal length. This is the case a chain of planar
# homographies cannot survive -- the outer frames meet the middle one's plane
# at a grazing angle -- and the one the cylinder exists for. The numbers are
# kept small so the whole sweep stitches in about a second.
SWEEP_FOCAL = 320.0
SWEEP_VIEW_WIDTH = 420
SWEEP_VIEW_HEIGHT = 320
SWEEP_YAWS = (-45.0, -15.0, 15.0, 45.0)


def _yaw_matrix(degrees):
    """Rotation of the camera about its vertical axis."""
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])


def _intrinsics(width, height, focal=SWEEP_FOCAL):
    """Camera matrix of a view of this size, principal point at its centre."""
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])


@pytest.fixture(scope="session")
def rotating_sweep(tmp_path_factory):
    """One 90 degree sweep: {paths, focal, span}, the frames left to right.

    The scene is a single very wide textured image, read as what the camera
    would see at yaw 0 if its sensor were that large; every frame is the part
    of it a camera turned by SWEEP_YAWS[k] sees, which is exactly the
    homography K R^T K^-1. Two frames are therefore related by a pure rotation
    about one centre -- a real sweep, not a set of crops.
    """
    # Wide enough to cover the outermost yaw plus that frame's own half angle,
    # and a quarter radian taller than one frame, so no view runs off the scene.
    half_width_angle = math.atan(SWEEP_VIEW_WIDTH / (2 * SWEEP_FOCAL))
    half_height_angle = math.atan(SWEEP_VIEW_HEIGHT / (2 * SWEEP_FOCAL))
    outermost = math.radians(max(abs(yaw) for yaw in SWEEP_YAWS))
    scene_width = 2 * int(SWEEP_FOCAL * math.tan(outermost + half_width_angle)) + 1
    scene_height = 2 * int(SWEEP_FOCAL * math.tan(half_height_angle + 0.25)) + 1
    scene = _textured_image(
        scene_width, scene_height, seed=7, blobs=scene_width * scene_height // 900
    )

    K_scene = _intrinsics(scene_width, scene_height)
    K_view = _intrinsics(SWEEP_VIEW_WIDTH, SWEEP_VIEW_HEIGHT)

    directory = tmp_path_factory.mktemp("sweep")
    paths = []
    for index, yaw in enumerate(SWEEP_YAWS, start=1):
        H = K_view @ _yaw_matrix(yaw).T @ np.linalg.inv(K_scene)
        view = cv2.warpPerspective(
            scene, H, (SWEEP_VIEW_WIDTH, SWEEP_VIEW_HEIGHT), flags=cv2.INTER_AREA
        )
        path = directory / f"sweep_{index}.jpg"
        cv2.imwrite(str(path), view)
        paths.append(path)
    return {
        "paths": paths,
        "focal": SWEEP_FOCAL,
        "span": max(SWEEP_YAWS) - min(SWEEP_YAWS),
        "size": (SWEEP_VIEW_WIDTH, SWEEP_VIEW_HEIGHT),
    }


@pytest.fixture(scope="session")
def vertical_pair_paths(tmp_path_factory):
    """Two crops of one tall scene, stacked: the top one and the bottom one."""
    scene = _textured_image(300, 700, seed=4, blobs=480)
    directory = tmp_path_factory.mktemp("vertical")
    paths = []
    for index, top in enumerate((0, 320)):
        path = directory / f"row_{index + 1}.jpg"
        cv2.imwrite(str(path), scene[top : top + 380].copy())
        paths.append(path)
    return paths


def _encode_multipart(fields, files):
    """Build a multipart/form-data body by hand.

    fields: {name: value}
    files: {name: (filename, content_bytes)} or {name: [(filename, content), ...]}
    for a field that repeats, which is how a whole image set is uploaded.
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
    for name, entries in files.items():
        if isinstance(entries, tuple):
            entries = [entries]
        for filename, content in entries:
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
