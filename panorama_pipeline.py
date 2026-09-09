from pathlib import Path

import cv2
import numpy as np

import birlestirme
import homografi
import matcher
from errors import PanoramaError
from homografi import MIN_MATCH_COUNT

# This is a library module: it never configures logging, the entry points
# (Step1_Sift.py, app.py) decide how log records are handled.
# PanoramaError is re-exported so callers keep importing it from here.
__all__ = [
    "EXAMPLES",
    "PROJECT_DIR",
    "PanoramaError",
    "example_paths",
    "get_example",
    "stitch_pair",
]


PROJECT_DIR = Path(__file__).resolve().parent

EXAMPLES = [
    {
        "id": "clock",
        "title": "Saat Kulesi",
        "folder": "images/Clock",
        "left": "sol1.jpg",
        "right": "sag1.jpg",
    },
    {
        "id": "school",
        "title": "Okul Bahcesi",
        "folder": "images/SchoolImage",
        "left": "sol2.jpg",
        "right": "sag2.jpg",
    },
    {
        "id": "street",
        "title": "Test Goruntusu",
        "folder": "images/test1",
        "left": "s1.jpg",
        "right": "s2.jpg",
    },
]


def get_example(example_id):
    for example in EXAMPLES:
        if example["id"] == example_id:
            return example
    raise PanoramaError("Secilen ornek veri bulunamadi.")


def example_paths(example_id):
    example = get_example(example_id)
    folder = PROJECT_DIR / example["folder"]
    return folder / example["left"], folder / example["right"]


def _read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise PanoramaError(f"Goruntu okunamadi: {path.name}")
    return image


def _detect_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) == 0:
        raise PanoramaError("Bu gorselde yeterli SIFT noktasi bulunamadi.")
    drawn = cv2.drawKeypoints(
        gray,
        keypoints,
        image.copy(),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return keypoints, descriptors, drawn


def _kucult(image, olcek):
    """Downscale with INTER_AREA, the right filter for shrinking."""
    yeni_genislik = max(1, int(round(image.shape[1] * olcek)))
    yeni_yukseklik = max(1, int(round(image.shape[0] * olcek)))
    return cv2.resize(image, (yeni_genislik, yeni_yukseklik), interpolation=cv2.INTER_AREA)


def _girisleri_olcekle(left, right, max_side):
    """Shrink both inputs by one common factor when either side is too large.

    Returns (left, right, scale); scale is 1.0 when nothing was resized.
    """
    if not max_side:
        return left, right, 1.0
    en_buyuk = max(left.shape[0], left.shape[1], right.shape[0], right.shape[1])
    if en_buyuk <= max_side:
        return left, right, 1.0
    olcek = float(max_side) / float(en_buyuk)
    return _kucult(left, olcek), _kucult(right, olcek), olcek


def _write_image(path, image):
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise PanoramaError(f"Cikti kaydedilemedi: {path.name}")


def stitch_pair(left_path, right_path, output_dir, max_side=None):
    """Run the full pipeline on one image pair and write the outputs.

    max_side: optional pixel cap on the longest side of the inputs. When the
    pair is larger, both images are downscaled by the same factor before
    detection (the applied factor is reported as "inputScale").
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    left = _read_image(left_path)
    right = _read_image(right_path)
    left, right, input_scale = _girisleri_olcekle(left, right, max_side)

    kp_left, des_left, left_keypoints = _detect_features(left)
    kp_right, des_right, right_keypoints = _detect_features(right)

    left_keypoints_path = output_dir / "sol_noktalar.jpg"
    right_keypoints_path = output_dir / "sag_noktalar.jpg"
    _write_image(left_keypoints_path, left_keypoints)
    _write_image(right_keypoints_path, right_keypoints)

    good_matches = matcher.match_features(kp_left, des_left, kp_right, des_right)
    if len(good_matches) < MIN_MATCH_COUNT:
        raise PanoramaError(
            f"Yeterli eslesme bulunamadi. Gerekli: {MIN_MATCH_COUNT}, bulunan: {len(good_matches)}"
        )

    match_preview = cv2.drawMatches(
        left,
        kp_left,
        right,
        kp_right,
        good_matches[:80],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    match_path = output_dir / "eslesme_final.jpg"
    _write_image(match_path, match_preview)

    ransac_path = output_dir / "ransac_temizlenmis_eslesmeler.jpg"
    H, mask = homografi.donusum_matrisi_hesapla(
        kp_left,
        kp_right,
        good_matches,
        left,
        right,
        ransac_cikti_yolu=str(ransac_path),
    )
    if H is None or mask is None:
        raise PanoramaError("Homografi hesaplanamadi; gorseller yeterince ortusmuyor olabilir.")

    panorama = birlestirme.panorama_birlestir(left, right, H)
    if panorama is None:
        raise PanoramaError("Panorama uretilemedi.")

    panorama_path = output_dir / "panorama_birlestirme.jpg"
    _write_image(panorama_path, panorama)

    inliers = int(np.sum(mask))
    return {
        "metrics": {
            "leftKeypoints": len(kp_left),
            "rightKeypoints": len(kp_right),
            "goodMatches": len(good_matches),
            "inliers": inliers,
            "panoramaWidth": int(panorama.shape[1]),
            "panoramaHeight": int(panorama.shape[0]),
            "inputScale": round(float(input_scale), 4),
        },
        "files": {
            "panorama": panorama_path.name,
            "leftKeypoints": left_keypoints_path.name,
            "rightKeypoints": right_keypoints_path.name,
            "matches": match_path.name,
            "ransac": ransac_path.name,
        },
    }
