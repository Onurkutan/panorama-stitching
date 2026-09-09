"""Command line entry point: stitch the bundled example image sets.

The pipeline itself lives in panorama_pipeline.py (shared with the web app);
this file only picks the image sets, decides where the outputs go, and shows
the last panorama on screen.

    python Step1_Sift.py                      # all sets -> outputs/<set>/
    python Step1_Sift.py --sets clock         # one set only
    python Step1_Sift.py --headless           # no GUI (also: HEADLESS=1)
    python Step1_Sift.py --out /tmp/panorama  # different output root
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2

import panorama_pipeline
from panorama_pipeline import EXAMPLES, PROJECT_DIR, PanoramaError, example_paths

# Avoid blocking in terminal/CI: HEADLESS=1 python Step1_Sift.py
HEADLESS = os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes")

SET_ADLARI = [ornek["id"] for ornek in EXAMPLES]


def _argumanlari_oku(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default="outputs",
        help="Output root directory; each set writes into <out>/<set>/ "
             "(relative paths are resolved against the project directory).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open any GUI window (env HEADLESS=1 does the same).",
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        choices=SET_ADLARI,
        default=SET_ADLARI,
        help="Subset of image sets to process (default: all).",
    )
    return parser.parse_args(argv)


def _cikti_koku(out):
    """Resolve --out against the project directory when it is relative."""
    yol = Path(out)
    if not yol.is_absolute():
        yol = PROJECT_DIR / yol
    return yol


def _bir_seti_isle(set_adi, cikti_koku):
    """Stitch one example set; returns the panorama path or None on failure."""
    sol_yol, sag_yol = example_paths(set_adi)
    set_cikti = cikti_koku / set_adi

    print(f"\n=== {set_adi} ({sol_yol.name} + {sag_yol.name}) ===")
    sonuc = panorama_pipeline.stitch_pair(sol_yol, sag_yol, set_cikti)

    olcumler = sonuc["metrics"]
    print(
        "Sol nokta: {leftKeypoints} | Sag nokta: {rightKeypoints} | "
        "Kaliteli eslesme: {goodMatches} | RANSAC saglam: {inliers} | "
        "Panorama: {panoramaWidth}x{panoramaHeight} | Olcek: {inputScale}".format(**olcumler)
    )
    panorama_yolu = set_cikti / sonuc["files"]["panorama"]
    print(f"Kaydedildi: {panorama_yolu}")
    return panorama_yolu


def main(argv=None):
    args = _argumanlari_oku(argv)
    # Entry point (not the library) decides how log records are shown.
    # stdout (not the default stderr) so log lines stay ordered with the prints.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    gorsel_goster = not (args.headless or HEADLESS)
    cikti_koku = _cikti_koku(args.out)

    son_panorama = None
    basarili = 0
    for set_adi in args.sets:
        try:
            son_panorama = _bir_seti_isle(set_adi, cikti_koku) or son_panorama
            basarili += 1
        except PanoramaError as hata:
            # One bad set must not stop the others.
            print(f"Bu set atlandi ({set_adi}): {hata}")

    print(f"\nIslem tamam! {basarili}/{len(args.sets)} goruntu seti islendi.")

    # Show only the last panorama to avoid window clutter.
    if gorsel_goster and son_panorama is not None:
        panorama = cv2.imread(str(son_panorama))
        if panorama is None:
            print(f"Panorama gosterilemedi: {son_panorama}")
            return 0
        cv2.imshow("Panorama (5. Asama)", panorama)
        print("Cikmak icin herhangi bir tusa bas.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


# --- Main entry point ---
if __name__ == "__main__":
    raise SystemExit(main())
