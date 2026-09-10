# panorama-stitching

[![CI](https://github.com/Onurkutan/panorama-stitching/actions/workflows/ci.yml/badge.svg)](https://github.com/Onurkutan/panorama-stitching/actions/workflows/ci.yml)

Two-image panorama stitching from scratch with Python and OpenCV: SIFT features,
FLANN matching with Lowe's ratio test, RANSAC homography, perspective warping,
exposure matching, seam feathering and automatic border cropping. Ships with a small
web UI (English / Turkish) that stitches your own photo pairs or the bundled demo pairs,
shows every intermediate step, and runs either against a tiny stdlib server or entirely
inside the browser.

| Clock tower | School yard | Pont du Gard |
| --- | --- | --- |
| ![Clock panorama](images/Clock/panorama_birlestirme.jpg) | ![School panorama](images/SchoolImage/panorama_birlestirme.jpg) | ![Aqueduct panorama](images/test1/panorama_birlestirme.jpg) |

## Live demo

[onurkutan.github.io/panorama-stitching](https://onurkutan.github.io/panorama-stitching/)
runs the very same Python modules inside the browser through
[Pyodide](https://pyodide.org) (CPython and OpenCV compiled to WebAssembly), so no server
is involved and uploaded photos never leave your machine. The first visit downloads about
20 MB of runtime; after that a pair stitches in a few seconds. Inputs are downscaled to
1400 px on the long side there. The page detects whether the Python server is available
and otherwise switches to the in-browser pipeline, so the same `index.html` and `static/`
serve both.

## Pipeline

1. **Feature detection** (`panorama_stitching/features.py`): grayscale conversion,
   `cv2.SIFT_create()`, keypoints and 128-d descriptors for both images.
2. **Matching** (`panorama_stitching/matching.py`): FLANN kd-tree kNN (k=2) and Lowe's
   ratio test (0.7).
3. **Homography** (`panorama_stitching/homography.py`): `cv2.findHomography` with RANSAC
   (reprojection threshold 5 px, minimum 10 matches), inlier visualization.
4. **Stitching** (`panorama_stitching/blending.py`): canvas size from the warped corners (degenerate
   homographies are rejected before anything is allocated), left image warped into the
   right image plane with eroded validity masks so the interpolation fringe never counts
   as content, per-channel exposure matching on the overlap, feather blending in a band
   around the seam that runs equidistant from both image borders (moving subjects stay
   unblended outside it), and trimming of the black borders.

Results on the bundled examples (FLANN and RANSAC are seeded, so the numbers are
reproducible; about 2 s per pair on a laptop CPU):

| Example | Keypoints L / R | Good matches | RANSAC inliers | Output size |
| --- | --- | --- | --- | --- |
| Clock tower | 8 313 / 5 516 | 1 571 | 1 490 | 1944 x 867 |
| School yard | 7 641 / 10 041 | 816 | 795 | 3440 x 1200 |
| Pont du Gard | 7 588 / 10 684 | 3 846 | 3 828 | 1812 x 696 |

## Quickstart

```bash
git clone https://github.com/Onurkutan/panorama-stitching.git
cd panorama-stitching
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt      # or: pip install -e ".[dev]"
```

### Web UI

```bash
python app.py            # optional port: python app.py 8080
```

Open `http://127.0.0.1:8000`. **Your photos** takes a left and a right photo (drag and
drop or file picker, about 30-50% overlap) and returns a downloadable panorama; **Demo
pairs** runs one of the bundled pairs with one click. Both show the SIFT keypoints, the
raw and the RANSAC-filtered matches and the keypoint / match / inlier counts. The
interface is available in English and Turkish (toggle in the header).
Generated files are written to `web_outputs/<job-id>/` and uploads to `web_uploads/`;
job folders older than a day are removed automatically. Uploads are capped at 25 MB and
inputs larger than 2400 px on the long side are downscaled before processing (the status
line says so).

### Command line

```bash
python -m panorama_stitching.cli --headless            # all three demo pairs
python -m panorama_stitching.cli --sets clock street   # a subset
python -m panorama_stitching.cli --out /tmp/pano       # different output root
```

(`panorama-stitch` is the same entry point after `pip install -e .`.) Each pair is
written to `outputs/<set>/` as `panorama.jpg`, `left_keypoints.jpg`,
`right_keypoints.jpg`, `matches.jpg` and `ransac_inliers.jpg`. Without `--headless`
(or `HEADLESS=1`) the last panorama is also shown in an OpenCV window. Failures such as
too few matches are reported per pair and the remaining pairs still run.

From Python:

```python
from panorama_stitching import stitch_pair

result = stitch_pair("left.jpg", "right.jpg", "out/", max_side=2400)
print(result["metrics"], result["files"])
```

## Development

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
pytest
```

## Project layout

```
panorama_stitching/     the package
  features.py             SIFT keypoints and descriptors
  matching.py             FLANN kNN matching + Lowe ratio test
  homography.py           RANSAC homography + inlier visualization
  blending.py             validity masks, exposure matching, seam feathering, auto-crop
  pipeline.py             stitch_pair(): the pipeline shared by the web app and the CLI
  cli.py                  command line entry point over the demo pairs
  errors.py               PanoramaError (with a stable .code) raised on unusable input
app.py                  stdlib HTTP server: static files, /api/examples, /api/stitch
index.html, static/     web UI (vanilla JS, no build step, English / Turkish)
tests/                  pytest suite (synthetic images plus one real downscaled pair)
images/                 demo image pairs and their stitched panoramas
```

## Limitations and ideas

- A single homography assumes a rotating camera or a distant scene; strong parallax
  (close objects with camera translation) cannot be aligned by any 3x3 warp.
- Blending is a feather band along a fixed geometric seam. Seam finding
  (`cv2.detail_DpSeamFinder`) or multi-band blending would hide misalignments better.
- Only two images are stitched; chaining more views needs a reference frame and bundle
  adjustment.

Requirements: Python 3.10+, `opencv-python` 4.8+ (SIFT is included in the main
package since 4.4), `numpy`. Errors raised by the pipeline are `PanoramaError`
instances with an English message and a stable `code` (for example
`not_enough_matches`), which the web UI maps to localized messages.

## Sample image credits

- `images/SchoolImage`: photographed by Onur Kutan.
- `images/Clock` (Carnegie Mellon University campus) and `images/test1` (Pont du Gard):
  third-party photographs used here for educational and demonstration purposes only.
  Copyright remains with their respective owners; they are not covered by this
  repository's license. Replace them with your own pairs if you reuse the project.

## Origin

The stitching pipeline started as a university team project
([expectation0/Panorama-Stitching-CV](https://github.com/expectation0/Panorama-Stitching-CV)).
This repository is maintained by Onur Kutan and continues that work with the web UI,
hardening, tests and tooling.

## License

MIT, see [LICENSE](LICENSE).
