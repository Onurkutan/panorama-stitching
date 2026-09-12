"""End-to-end pipeline: two overlapping photos in, one panorama out.

This is a library module: it never configures logging, the entry points
(panorama_stitching.cli, app.py) decide how log records are handled.
"""

import logging
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import blending, homography, matching
from .errors import PanoramaError
from .features import detect_features
from .homography import MIN_MATCH_COUNT

# By name rather than as a module: `projection` is also stitch_set's parameter.
from .projection import estimate_focal, warp_cylindrical

__all__ = [
    "EXAMPLES",
    "MAX_IMAGES",
    "MIN_IMAGES",
    "MIN_PAIR_INLIERS",
    "PROJECTIONS",
    "PROJECT_DIR",
    "PanoramaError",
    "example_paths",
    "get_example",
    "stitch_pair",
    "stitch_set",
]

_logger = logging.getLogger(__name__)


# The repository root: this file lives in <root>/panorama_stitching/.
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Bundled demo sets: three pairs and one six-photo sweep. "files" are listed
# in capture order for readability only; the pipeline recovers the layout.
EXAMPLES = [
    {
        "id": "clock",
        "title": "Clock tower",
        "folder": "images/Clock",
        "files": ["sol1.jpg", "sag1.jpg"],
    },
    {
        "id": "school",
        "title": "School yard",
        "folder": "images/SchoolImage",
        "files": ["sol2.jpg", "sag2.jpg"],
    },
    {
        "id": "street",
        "title": "Pont du Gard",
        "folder": "images/test1",
        "files": ["s1.jpg", "s2.jpg"],
    },
    {
        "id": "balcony",
        "title": "Balcony sweep",
        "folder": "images/BalconySweep",
        "files": [f"photo_{index}.jpg" for index in range(1, 7)],
    },
]


def get_example(example_id):
    """Return the EXAMPLES entry with this id, or raise PanoramaError."""
    for example in EXAMPLES:
        if example["id"] == example_id:
            return example
    raise PanoramaError("The selected example set was not found.", code="example_not_found")


def example_paths(example_id):
    """Absolute image paths of one bundled example set, as a tuple.

    Pairs unpack as ``left, right = example_paths("clock")``; the sweep has six.
    """
    example = get_example(example_id)
    folder = PROJECT_DIR / example["folder"]
    return tuple(folder / name for name in example["files"])


# HEIC/HEIF (the default iPhone format) is not decoded by OpenCV. It is read
# through the optional pillow-heif package when it is installed; the file is
# recognised by extension or by the ISO base media "ftyp" brand.
HEIC_SUFFIXES = {".heic", ".heif", ".hif"}
_HEIC_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"}


def looks_like_heic(path):
    """True when the file has a HEIC/HEIF extension or ftyp brand."""
    path = Path(path)
    if path.suffix.lower() in HEIC_SUFFIXES:
        return True
    try:
        with open(path, "rb") as handle:
            header = handle.read(16)
    except OSError:
        return False
    return len(header) >= 12 and header[4:8] == b"ftyp" and header[8:12] in _HEIC_BRANDS


def _decode_heic(path):
    """Decode a HEIC file to a BGR array with pillow-heif, or explain what is missing."""
    try:
        import pillow_heif
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise PanoramaError(
            f"{Path(path).name} is a HEIC photo; decoding it needs the optional pillow-heif "
            'package (pip install "panorama-stitching[heic]" or pip install pillow-heif), '
            "or export the photo as JPEG.",
            code="heic_unsupported",
        ) from exc

    pillow_heif.register_heif_opener()
    try:
        with Image.open(path) as picture:
            # exif_transpose applies the stored orientation the way cv2.imread does for JPEG.
            rgb = ImageOps.exif_transpose(picture).convert("RGB")
            array = np.asarray(rgb)
    except Exception as exc:
        raise PanoramaError(
            f"Could not decode the HEIC photo: {Path(path).name}", code="image_unreadable"
        ) from exc
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def read_image(path):
    """Read an image as BGR; JPEG/PNG/... through OpenCV, HEIC through pillow-heif.

    Raises PanoramaError(code="image_unreadable") when the file cannot be
    decoded and code="heic_unsupported" when a HEIC file is given but the
    optional decoder is not installed.
    """
    image = cv2.imread(str(path))
    if image is not None:
        return image
    if looks_like_heic(path):
        return _decode_heic(path)
    # Path() so a plain string path also produces a message, not a crash.
    raise PanoramaError(f"Could not read the image: {Path(path).name}", code="image_unreadable")


def _read_image(path):
    return read_image(path)


def _downscale(image, scale):
    """Downscale with INTER_AREA, the right filter for shrinking."""
    new_width = max(1, int(round(image.shape[1] * scale)))
    new_height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _scale_images(images, max_side):
    """Shrink a whole set by ONE common factor when any side is too large.

    One shared factor (rather than per-image ones) keeps the relative scale of
    the photos intact, which is what the homographies between them assume.
    Returns (images, scale); scale is 1.0 when nothing was resized.
    """
    images = list(images)
    if not max_side:
        return images, 1.0
    longest_side = max(max(image.shape[0], image.shape[1]) for image in images)
    if longest_side <= max_side:
        return images, 1.0
    scale = float(max_side) / float(longest_side)
    return [_downscale(image, scale) for image in images], scale


def _scale_inputs(left, right, max_side):
    """Two-image case of _scale_images, kept for stitch_pair's signature."""
    images, scale = _scale_images([left, right], max_side)
    return images[0], images[1], scale


def _write_image(path, image):
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise PanoramaError(f"Could not save the output: {path.name}", code="output_write_failed")


# Stages reported through the optional `progress` callback of stitch_pair, in
# order. Clients (the web worker, a CLI spinner) can map them onto a bar.
PROGRESS_STAGES = ("features", "matching", "homography", "blending", "saving")


def _report(progress, stage):
    """Call progress(stage, step, total) when a callback was given."""
    if progress is not None:
        progress(stage, PROGRESS_STAGES.index(stage) + 1, len(PROGRESS_STAGES))


def stitch_pair(left_path, right_path, output_dir, max_side=None, progress=None):
    """Run the full pipeline on one image pair and write the outputs.

    max_side: optional pixel cap on the longest side of the inputs. When the
    pair is larger, both images are downscaled by the same factor before
    detection (the applied factor is reported as "inputScale").
    progress: optional callable(stage, step, total) invoked before each stage
    listed in PROGRESS_STAGES; errors raised by it propagate to the caller.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    left = _read_image(left_path)
    right = _read_image(right_path)
    left, right, input_scale = _scale_inputs(left, right, max_side)

    _report(progress, "features")
    kp_left, des_left, left_keypoints = detect_features(left)
    kp_right, des_right, right_keypoints = detect_features(right)

    left_keypoints_path = output_dir / "left_keypoints.jpg"
    right_keypoints_path = output_dir / "right_keypoints.jpg"
    _write_image(left_keypoints_path, left_keypoints)
    _write_image(right_keypoints_path, right_keypoints)

    _report(progress, "matching")
    good_matches = matching.match_features(kp_left, des_left, kp_right, des_right)
    if len(good_matches) < MIN_MATCH_COUNT:
        raise PanoramaError(
            f"Not enough matches. Required: {MIN_MATCH_COUNT}, found: {len(good_matches)}",
            code="not_enough_matches",
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
    match_path = output_dir / "matches.jpg"
    _write_image(match_path, match_preview)

    _report(progress, "homography")
    ransac_path = output_dir / "ransac_inliers.jpg"
    H, mask = homography.estimate_homography(
        kp_left,
        kp_right,
        good_matches,
        left,
        right,
        inlier_output_path=str(ransac_path),
    )
    if H is None or mask is None:
        raise PanoramaError(
            "Could not compute the homography; the images may not overlap enough.",
            code="homography_failed",
        )

    _report(progress, "blending")
    panorama = blending.stitch_images(left, right, H)
    if panorama is None:
        raise PanoramaError("The panorama could not be produced.", code="unexpected")

    _report(progress, "saving")
    panorama_path = output_dir / "panorama.jpg"
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


# ------------------------------------------------------------------ image sets

# The smallest and largest set the pipeline accepts. Every photo is matched
# against every other one, so the matching work grows as N^2: six photos
# already mean fifteen matching runs, about as much as one request should cost.
MIN_IMAGES = 2
MAX_IMAGES = 6

# A pair becomes an edge of the arrangement graph only when it clears the very
# same gate a two-image stitch has to clear.
MIN_PAIR_INLIERS = MIN_MATCH_COUNT

# The surface every photo is warped onto before they are composited.
#   planar       -- chained homographies, the classic two-image stitch
#   cylindrical  -- every photo projected onto a cylinder first (see projection)
#   auto         -- planar for a pair, cylindrical from three photos on, where
#                   the sweep is wide enough for the planar chain to blow up
PROJECTIONS = ("auto", "planar", "cylindrical")

# How far a cylinder image's own validity mask is shrunk before SIFT runs on it.
# A keypoint is described from a patch around it, so one sitting right on the
# border of the projected content is described half from the black corner
# outside it; those are not the same features the neighbouring photo sees.
FEATURE_MASK_MARGIN = 8


def _resolve_projection(projection, count):
    """Turn the requested projection into the one this set will actually use."""
    if projection not in PROJECTIONS:
        listed = ", ".join(PROJECTIONS)
        raise PanoramaError(
            f"Unknown projection: {projection!r}. Use one of: {listed}.",
            code="form_invalid",
        )
    if projection != "auto":
        return projection
    # Two photos are one hop apart, so nothing can accumulate and the planar
    # stitch stays the sharpest option; three or more mean a sweep.
    return "planar" if count <= 2 else "cylindrical"


def _set_progress_total(count, projection="planar"):
    """Steps a stitch_set run reports.

        count                      "features", one per image
      + count * (count - 1) // 2   "matching", one per pair
      + count - 1                  "matching" again, on the cylinder: the tree
                                   edges are re-matched on the warped images
                                   (cylindrical projection only)
      + 3                          "homography", "blending", "saving"

    A cylindrical run that has to fall back to the planar chain (see
    stitch_set) stops short of the total it announced -- the steps it skips are
    the tree edges it never got to re-match.
    """
    steps = count + count * (count - 1) // 2 + 3
    if projection == "cylindrical":
        steps += count - 1
    return steps


def _set_reporter(progress, total):
    """Return report(stage) -> progress(stage, step, total) with a running step.

    stitch_pair has exactly one step per stage, so PROGRESS_STAGES.index() is
    enough there. A set repeats "features" and "matching", so the step number
    has to be counted rather than derived from the stage name.
    """
    state = {"step": 0}

    def report(stage):
        state["step"] += 1
        if progress is not None:
            progress(stage, state["step"], total)

    return report


def _read_images(paths):
    """Read every input image, naming the offending one when a read fails."""
    images = []
    for index, path in enumerate(paths, start=1):
        try:
            images.append(read_image(path))
        except PanoramaError as exc:
            raise PanoramaError(
                f"Image {index}: {exc}", code=exc.code, details={"image": index}
            ) from exc
    return images


def _detect_all(images, output_dir, report):
    """SIFT features for every image, plus its keypoints_<k>.jpg overlay."""
    keypoints, descriptors, names = [], [], []
    for index, image in enumerate(images, start=1):
        report("features")
        try:
            image_keypoints, image_descriptors, drawn = detect_features(image)
        except PanoramaError as error:
            raise PanoramaError(
                f"Image {index}: {error}", code=error.code, details={"image": index}
            ) from error
        name = f"keypoints_{index}.jpg"
        _write_image(output_dir / name, drawn)
        keypoints.append(image_keypoints)
        descriptors.append(image_descriptors)
        names.append(name)
    return keypoints, descriptors, names


def _evaluate_pairs(images, keypoints, descriptors, report):
    """Match every pair and keep the ones that survive RANSAC as graph edges.

    A pair that does not match is not an error here: two photos of one row
    simply need not overlap, and the arrangement is recovered from the pairs
    that do. Each edge carries the homography mapping image i into image j's
    plane, together with the counts the UI reports.
    """
    edges = []
    count = len(images)
    for i in range(count):
        for j in range(i + 1, count):
            report("matching")
            try:
                good = matching.match_features(
                    keypoints[i], descriptors[i], keypoints[j], descriptors[j]
                )
                if len(good) < MIN_MATCH_COUNT:
                    continue
                H, mask = homography.estimate_homography(
                    keypoints[i], keypoints[j], good, images[i], images[j]
                )
                if H is None or mask is None:
                    continue
                # A homography that could never produce a canvas is no better
                # than a pair that did not match at all.
                blending._validate_homography(H)
            except PanoramaError:
                continue
            inliers = int(np.sum(mask))
            if inliers < MIN_PAIR_INLIERS:
                continue
            edges.append({"i": i, "j": j, "H": H, "matches": good, "inliers": inliers})
    return edges


def _maximum_spanning_tree(count, edges):
    """Kruskal over the edges, strongest (most inliers) first.

    The tree is the skeleton of the arrangement: at most N-1 pairs joining the
    photos through their most reliable overlaps, the same idea OpenCV's
    Stitcher uses. Ties are broken by input order so the result is stable.
    """
    parent = list(range(count))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    tree = []
    for edge in sorted(edges, key=lambda item: (-item["inliers"], item["i"], item["j"])):
        root_i, root_j = find(edge["i"]), find(edge["j"])
        if root_i == root_j:
            continue
        parent[root_i] = root_j
        tree.append(edge)
        if len(tree) == count - 1:
            break
    return tree


def _tree_adjacency(count, tree):
    """Undirected neighbour lists of the spanning tree."""
    adjacency = [[] for _ in range(count)]
    for edge in tree:
        adjacency[edge["i"]].append(edge["j"])
        adjacency[edge["j"]].append(edge["i"])
    return adjacency


def _breadth_first(adjacency, start):
    """Nodes reachable from `start`, in breadth-first order."""
    order = [start]
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbour in adjacency[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                order.append(neighbour)
                queue.append(neighbour)
    return order


def _require_connected(count, tree):
    """Raise image_not_connected for the first photo the tree leaves out."""
    if len(tree) == count - 1:
        return

    adjacency = _tree_adjacency(count, tree)
    components = []
    seen = set()
    for node in range(count):
        if node in seen:
            continue
        component = _breadth_first(adjacency, node)
        seen.update(component)
        components.append(component)

    # The largest component is the panorama; the first photo outside it is the
    # one the user has to replace (ties go to the earliest component).
    main = max(components, key=lambda component: (len(component), -min(component)))
    outside = min(set(range(count)) - set(main))
    raise PanoramaError(
        f"Image {outside + 1} does not overlap any of the other photos.",
        code="image_not_connected",
        details={"image": outside + 1},
    )


def _tree_centre(count, adjacency):
    """The tree node with the smallest eccentricity.

    Blending outward from the centre keeps the longest chain of composed
    homographies as short as possible, so accumulated warp error stays small.
    Ties go to the later photo, which makes the second image the reference of
    a two-image set -- exactly the plane stitch_pair warps the first one into.
    """
    best_node, best_eccentricity = 0, None
    for start in range(count):
        distance = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbour in adjacency[node]:
                if neighbour not in distance:
                    distance[neighbour] = distance[node] + 1
                    queue.append(neighbour)
        eccentricity = max(distance.values())
        if best_eccentricity is None or eccentricity <= best_eccentricity:
            best_node, best_eccentricity = start, eccentricity
    return best_node


def _compose_to_reference(count, tree, reference):
    """Chain the pairwise homographies into H_k -> reference for every image.

    Returns (homographies, order); `order` is the breadth-first walk of the
    tree from the reference, which is also the order the images are blended in
    -- each of them then meets a composite it genuinely overlaps.
    """
    # step[(a, b)] maps image a into image b's plane. An edge is stored in the
    # i -> j direction, so walking it backwards means inverting it.
    step = {}
    for edge in tree:
        step[(edge["i"], edge["j"])] = edge["H"]
        step[(edge["j"], edge["i"])] = np.linalg.inv(edge["H"])

    adjacency = _tree_adjacency(count, tree)
    to_reference = [None] * count
    to_reference[reference] = np.eye(3)
    order = [reference]
    queue = deque([reference])
    while queue:
        parent = queue.popleft()
        for node in adjacency[parent]:
            if to_reference[node] is not None:
                continue
            hop = step[(node, parent)]
            # H_node->reference = H_parent->reference @ H_node->parent; the
            # reference's own entry is the identity, so skip that multiply.
            to_reference[node] = hop if parent == reference else to_reference[parent] @ hop
            order.append(node)
            queue.append(node)
    return to_reference, order


def _left_to_right(images, to_reference):
    """1-based image indices sorted by their warped centre's x coordinate.

    This is the arrangement the graph recovered: the photos were handed over
    in any order, and this is the order they actually appear in.
    """
    centres = []
    for index, (image, H) in enumerate(zip(images, to_reference, strict=True)):
        height, width = image.shape[:2]
        centre = np.float32([[[width / 2.0, height / 2.0]]])
        warped = cv2.perspectiveTransform(centre, np.asarray(H, dtype=np.float64))
        centres.append((float(warped[0, 0, 0]), index))
    centres.sort()
    return [index + 1 for _x, index in centres]


def _write_pair_previews(images, keypoints, tree, output_dir):
    """Write matches_<i>_<j>.jpg and ransac_<i>_<j>.jpg for the tree edges.

    Only the pairs that ended up in the tree get a preview: those are the
    overlaps the panorama is actually built from, and drawing all N(N-1)/2 of
    them would bury those among the pairs that were rejected.
    """
    matches_names, ransac_names, pairs, pair_metrics = [], [], [], []
    for edge in tree:
        i, j = edge["i"], edge["j"]
        good = edge["matches"]

        preview = cv2.drawMatches(
            images[i],
            keypoints[i],
            images[j],
            keypoints[j],
            good[:80],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        matches_name = f"matches_{i + 1}_{j + 1}.jpg"
        _write_image(output_dir / matches_name, preview)

        # Re-running RANSAC is what draws the inlier overlay; it is seeded, so
        # it reproduces the very same model the edge already carries.
        ransac_name = f"ransac_{i + 1}_{j + 1}.jpg"
        homography.estimate_homography(
            keypoints[i],
            keypoints[j],
            good,
            images[i],
            images[j],
            inlier_output_path=str(output_dir / ransac_name),
        )

        matches_names.append(matches_name)
        ransac_names.append(ransac_name)
        pairs.append([i + 1, j + 1])
        pair_metrics.append(
            {"images": [i + 1, j + 1], "goodMatches": len(good), "inliers": edge["inliers"]}
        )
    return matches_names, ransac_names, pairs, pair_metrics


def _cylindrical_chain(images, tree, reference, report):
    """Redo the tree edges on cylindrically projected copies of the photos.

    The arrangement is already known at this point, so only the N-1 edges of
    the spanning tree are matched again -- not all N(N-1)/2 pairs. Their
    homographies come out close to translations (see projection), which is what
    lets them be chained over a wide sweep at all.

    Returns (images, masks, homographies, order, focal): the cylinder images
    with their validity masks, each one's homography into the reference
    cylinder, the breadth-first blending order, and the focal length the
    cylinder was built with. Raises PanoramaError as soon as anything about the
    cylinder does not work out -- too few matches on an edge after the warp, a
    homography that would not produce a canvas -- so that stitch_set can fall
    back to the planar chain.
    """
    count = len(images)
    height, width = images[reference].shape[:2]
    focal = estimate_focal([edge["H"] for edge in tree], width, height)

    warped, masks, keypoints, descriptors = [], [], [], []
    for image in images:
        cylinder, mask = warp_cylindrical(image, focal)
        points, point_descriptors, _drawn = detect_features(
            cylinder, mask=blending._erode_mask(mask, FEATURE_MASK_MARGIN)
        )
        warped.append(cylinder)
        masks.append(mask)
        keypoints.append(points)
        descriptors.append(point_descriptors)

    edges = []
    for edge in tree:
        report("matching")
        i, j = edge["i"], edge["j"]
        good = matching.match_features(keypoints[i], descriptors[i], keypoints[j], descriptors[j])
        H, mask = homography.estimate_homography(
            keypoints[i], keypoints[j], good, warped[i], warped[j]
        )
        blending._validate_homography(H)
        inliers = int(np.sum(mask))
        if inliers < MIN_PAIR_INLIERS:
            raise PanoramaError(
                f"Images {i + 1} and {j + 1} no longer overlap once projected "
                f"onto the cylinder: {inliers} inliers.",
                code="not_enough_matches",
                details={"pair": [i + 1, j + 1]},
            )
        edges.append({"i": i, "j": j, "H": H})

    to_reference, order = _compose_to_reference(count, edges, reference)
    # Ask for the canvas before anything is allocated: a cylinder that is still
    # not flat enough fails here, in time for the planar chain to get its turn.
    blending._compute_canvas_set(warped, to_reference)
    return warped, masks, to_reference, order, focal


def stitch_set(paths, output_dir, max_side=None, progress=None, projection="auto"):
    """Stitch 2..MAX_IMAGES overlapping photos handed over in any order.

    Nothing about the arrangement is assumed. Every pair is matched, the pairs
    that survive RANSAC form a graph weighted by inlier count, and its maximum
    spanning tree is the skeleton of the panorama. The centre of that tree
    becomes the reference plane, every other photo's homography is chained
    along its tree path, and the images are blended outward from the centre.
    Because the skeleton is a tree and not a chain, a vertical strip or a 2x2
    block stitches exactly like a left-to-right row.

    From three photos on, the photos are projected onto a cylinder before they
    are composited (see projection): a sweep wide enough to need three shots is
    wide enough for a chain of planar homographies to blow up. The arrangement
    is recovered on the unprojected photos either way, so only the N-1 tree
    edges are matched a second time, on the cylinder images. If any of them
    fails there the run falls back to the planar chain and says so in
    metrics["projection"]; the previews (keypoints, matches, RANSAC inliers)
    always come from the first, unprojected stage, which is the one whose
    numbers the metrics report.

    max_side: optional pixel cap on the longest side of any input; the whole
    set is then scaled by one common factor (reported as "inputScale").
    progress: optional callable(stage, step, total), called once per image
    during "features", once per pair during "matching" (plus once per tree edge
    on the cylindrical path), and once each for "homography", "blending" and
    "saving". See _set_progress_total for the step count.
    projection: "auto" (planar for two photos, cylindrical from three on),
    "planar" or "cylindrical".
    """
    paths = list(paths)
    count = len(paths)
    if count < MIN_IMAGES:
        raise PanoramaError(
            f"A panorama needs at least {MIN_IMAGES} photos; {count} were given.",
            code="too_few_images",
        )
    if count > MAX_IMAGES:
        raise PanoramaError(
            f"At most {MAX_IMAGES} photos can be stitched at once; {count} were given.",
            code="too_many_images",
        )

    requested = _resolve_projection(projection, count)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = _set_reporter(progress, _set_progress_total(count, requested))

    images, input_scale = _scale_images(_read_images(paths), max_side)
    keypoints, descriptors, keypoint_files = _detect_all(images, output_dir, report)
    edges = _evaluate_pairs(images, keypoints, descriptors, report)

    tree = _maximum_spanning_tree(count, edges)
    _require_connected(count, tree)
    reference = _tree_centre(count, _tree_adjacency(count, tree))

    cylinder = None
    if requested == "cylindrical":
        try:
            cylinder = _cylindrical_chain(images, tree, reference, report)
        except PanoramaError as error:
            # The arrangement stands either way, so a cylinder that does not
            # work out costs the run nothing but the time it took.
            _logger.info("Cylindrical projection failed (%s); using the planar chain.", error)

    report("homography")
    used = "planar"
    focal = None
    blend_images, blend_masks = images, None
    # Composed before the sort below: the order the tree edges are in decides
    # the breadth-first blending order, and this is the one the planar path has
    # always used.
    to_reference, blend_order = _compose_to_reference(count, tree, reference)
    if cylinder is not None:
        blend_images, blend_masks, to_reference, blend_order, focal = cylinder
        used = "cylindrical"

    # The previews are listed in input order, which is how the UI labels them.
    tree.sort(key=lambda edge: (edge["i"], edge["j"]))
    matches_files, ransac_files, pairs, pair_metrics = _write_pair_previews(
        images, keypoints, tree, output_dir
    )

    report("blending")
    panorama = blending.stitch_set_images(
        blend_images, to_reference, reference, blend_order, masks=blend_masks
    )
    if panorama is None:
        raise PanoramaError("The panorama could not be produced.", code="unexpected")

    report("saving")
    panorama_path = output_dir / "panorama.jpg"
    _write_image(panorama_path, panorama)

    return {
        "metrics": {
            "imageCount": count,
            "keypoints": [len(points) for points in keypoints],
            "pairs": pair_metrics,
            "pairsEvaluated": count * (count - 1) // 2,
            "reference": reference + 1,
            "order": _left_to_right(blend_images, to_reference),
            "projection": used,
            "focalPx": None if focal is None else round(float(focal), 2),
            "totalKeypoints": sum(len(points) for points in keypoints),
            "totalGoodMatches": sum(pair["goodMatches"] for pair in pair_metrics),
            "totalInliers": sum(pair["inliers"] for pair in pair_metrics),
            "panoramaWidth": int(panorama.shape[1]),
            "panoramaHeight": int(panorama.shape[0]),
            "inputScale": round(float(input_scale), 4),
        },
        "files": {
            "panorama": panorama_path.name,
            "keypoints": keypoint_files,
            "matches": matches_files,
            "ransac": ransac_files,
            "pairs": pairs,
        },
    }
