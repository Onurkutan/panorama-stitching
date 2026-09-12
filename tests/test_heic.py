"""Optional HEIC input: detection, decoding through pillow-heif, and the message without it."""

import builtins
import sys

import cv2
import numpy as np
import pytest

from panorama_stitching.errors import PanoramaError
from panorama_stitching.pipeline import looks_like_heic, read_image

pillow_heif = pytest.importorskip("pillow_heif")


def _write_heic(path, size=(96, 64)):
    """Encode a small gradient as HEIC; skip the test when the wheel has no encoder."""
    from PIL import Image

    pillow_heif.register_heif_opener()
    width, height = size
    rgb = np.zeros((height, width, 3), np.uint8)
    rgb[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    rgb[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    rgb[..., 2] = 200
    try:
        Image.fromarray(rgb).save(path, format="HEIF", quality=90)
    except Exception as exc:  # pragma: no cover - depends on the wheel's codecs
        pytest.skip(f"HEIF encoder unavailable: {exc}")
    return rgb


def test_looks_like_heic_by_extension_and_brand(tmp_path):
    by_name = tmp_path / "photo.HEIC"
    by_name.write_bytes(b"not really")
    assert looks_like_heic(by_name)

    by_brand = tmp_path / "photo.bin"
    by_brand.write_bytes(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic")
    assert looks_like_heic(by_brand)

    jpeg = tmp_path / "photo.jpg"
    cv2.imwrite(str(jpeg), np.zeros((8, 8, 3), np.uint8))
    assert not looks_like_heic(jpeg)


def test_read_image_decodes_heic(tmp_path):
    path = tmp_path / "gradient.heic"
    rgb = _write_heic(path)

    image = read_image(path)

    assert image.shape == rgb.shape
    # BGR order and lossy compression: the blue channel is the constant one.
    assert abs(int(image[32, 48, 0]) - 200) < 12
    assert abs(int(image[32, 90, 2]) - int(rgb[32, 90, 0])) < 20


def test_read_image_heic_without_decoder_explains(tmp_path, monkeypatch):
    path = tmp_path / "photo.heic"
    path.write_bytes(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 64)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pillow_heif":
            raise ImportError("no module named pillow_heif")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "pillow_heif", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(PanoramaError) as info:
        read_image(path)
    assert info.value.code == "heic_unsupported"
    assert "pillow-heif" in str(info.value)


def test_read_image_unreadable_non_heic(tmp_path):
    path = tmp_path / "garbage.jpg"
    path.write_bytes(b"definitely not an image")
    with pytest.raises(PanoramaError) as info:
        read_image(path)
    assert info.value.code == "image_unreadable"
