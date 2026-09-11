import json
import logging
import mimetypes
import re
import shutil
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2

from panorama_stitching.errors import PanoramaError
from panorama_stitching.pipeline import (
    EXAMPLES,
    MAX_IMAGES,
    MIN_IMAGES,
    PROJECT_DIR,
    example_paths,
    stitch_set,
)

STATIC_DIR = PROJECT_DIR / "static"
OUTPUT_DIR = PROJECT_DIR / "web_outputs"
UPLOAD_DIR = PROJECT_DIR / "web_uploads"

# Only these top-level segments may be served under /media/<segment>/...
MEDIA_ROOTS = {
    "images": PROJECT_DIR / "images",
    "web_outputs": OUTPUT_DIR,
    "web_uploads": UPLOAD_DIR,
}

MAX_UPLOAD_BYTES = 40 * 1024 * 1024  # 40 MB, enough for a set of phone photos
JOB_TTL_HOURS = 24

# Longest side (in pixels) the web pipeline works on. Larger inputs are
# downscaled by a common factor before detection: phone photos of 4000 px or
# more would otherwise cost many seconds of SIFT time and hundreds of MB per
# request for no visible gain in the browser.
MAX_INPUT_SIDE = 2400

CONNECTION_ERRORS = (ConnectionAbortedError, BrokenPipeError, ConnectionResetError)


def _safe_join(root, relative_path):
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError("Invalid path") from None
    if candidate.is_dir():
        raise ValueError("Invalid path")
    return candidate


def _sweep_job_dirs():
    """Best-effort removal of job directories older than JOB_TTL_HOURS."""
    cutoff = time.time() - JOB_TTL_HOURS * 3600
    for base in (OUTPUT_DIR, UPLOAD_DIR):
        try:
            entries = list(base.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir() and entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
            except OSError:
                continue


def _json_response(handler, payload, status=200):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("X-Content-Type-Options", "nosniff")
        handler.end_headers()
        handler.wfile.write(body)
    except CONNECTION_ERRORS:
        handler.close_connection = True


def _parse_disposition(header_value):
    values = {}
    for key, value in re.findall(r'(\w+)="([^"]*)"', header_value):
        values[key] = value
    return values


def _upload_list(files, name):
    """The uploads of one multipart field, always as a list.

    _read_multipart keeps a field that appears once as a single entry, so a
    form carrying exactly one `images` file still reads as a one-element list
    here while `mode` and the other scalar fields keep their old shape.
    """
    upload = files.get(name)
    if upload is None:
        return []
    if isinstance(upload, list):
        return upload
    return [upload]


def _media_url(path):
    """A /media/ URL for a file that lives under one of the media roots.

    Built from the roots rather than from PROJECT_DIR: an upload directory is
    configurable (the tests point it at a temporary path), and only the roots
    know where it currently is.
    """
    for name, root in MEDIA_ROOTS.items():
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        return f"/media/{name}/{relative.as_posix()}"
    return None


def _job_urls(job_id, files):
    """Map the pipeline's output file names onto /generated/<job>/ URLs.

    "panorama" is a single name, "keypoints"/"matches"/"ransac" are lists, and
    "pairs" is not a file list at all but the [i, j] labels belonging to them.
    """
    urls = {}
    for name, value in files.items():
        if name == "pairs":
            urls[name] = value
        elif isinstance(value, list):
            urls[name] = [f"/generated/{job_id}/{item}" for item in value]
        else:
            urls[name] = f"/generated/{job_id}/{value}"
    return urls


class PanoramaHandler(BaseHTTPRequestHandler):
    # Socket-level timeout so a client that opens a connection and stalls
    # (e.g. lying Content-Length, never sends the rest of the body) cannot
    # pin a worker thread forever.
    timeout = 30

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/":
            return self._serve_file(PROJECT_DIR / "index.html")

        if path == "/api/examples":
            return _json_response(self, {"examples": self._examples_payload()})

        if path.startswith("/static/"):
            try:
                target = _safe_join(STATIC_DIR, path.removeprefix("/static/"))
            except ValueError:
                return self._not_found()
            return self._serve_file(target)

        if path.startswith("/media/"):
            return self._serve_media(path.removeprefix("/media/"))

        # The page uses site-relative image paths so it can also be served as a
        # static site (GitHub Pages); map them onto the images media root.
        if path.startswith("/images/"):
            return self._serve_media(path.removeprefix("/"))

        if path.startswith("/generated/"):
            try:
                target = _safe_join(OUTPUT_DIR, path.removeprefix("/generated/"))
            except ValueError:
                return self._not_found()
            return self._serve_file(target, cache_control="no-store")

        return self._not_found()

    def _serve_media(self, rel_path):
        root_name, _sep, rest = rel_path.partition("/")
        root = MEDIA_ROOTS.get(root_name)
        if root is None:
            return self._not_found()
        try:
            target = _safe_join(root, rest)
        except ValueError:
            return self._not_found()
        cache_control = "no-store" if root_name == "web_uploads" else None
        return self._serve_file(target, cache_control=cache_control)

    def do_POST(self):
        if urlparse(self.path).path != "/api/stitch":
            return self._not_found()

        _sweep_job_dirs()

        content_length_header = self.headers.get("Content-Length")
        try:
            content_length = int(content_length_header)
            if content_length < 0:
                raise ValueError("negative content length")
        except (TypeError, ValueError):
            self.close_connection = True
            return _json_response(
                self,
                {
                    "ok": False,
                    "error": "The content length header is missing or invalid.",
                    "code": "form_invalid",
                },
                status=400,
            )

        if content_length > MAX_UPLOAD_BYTES:
            self.close_connection = True
            return _json_response(
                self,
                {
                    "ok": False,
                    "error": (
                        "The uploaded data is too large "
                        f"({MAX_UPLOAD_BYTES // (1024 * 1024)} MB maximum)."
                    ),
                    "code": "upload_too_large",
                },
                status=413,
            )

        try:
            fields, files = self._read_multipart(content_length)
            job_id = uuid.uuid4().hex[:12]
            job_dir = OUTPUT_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            mode = fields.get("mode", "example")
            if mode == "example":
                # The bundled examples are pairs, but they run through the very
                # same set pipeline so both modes answer with one shape.
                image_paths = list(example_paths(fields.get("example", "clock")))
            else:
                upload_dir = UPLOAD_DIR / job_id
                upload_dir.mkdir(parents=True, exist_ok=True)
                image_paths = self._save_uploads(files, upload_dir)

            source = [_media_url(path) for path in image_paths]
            result = stitch_set(image_paths, job_dir, max_side=MAX_INPUT_SIDE)
            return _json_response(
                self,
                {
                    "ok": True,
                    "jobId": job_id,
                    "source": source,
                    "metrics": result["metrics"],
                    "files": _job_urls(job_id, result["files"]),
                },
            )
        except PanoramaError as exc:
            payload = {"ok": False, "error": str(exc), "code": exc.code}
            if exc.details is not None:
                payload["details"] = exc.details
            return _json_response(self, payload, status=422)
        except Exception:
            logging.exception("[web] unexpected error while handling /api/stitch")
            return _json_response(
                self,
                {
                    "ok": False,
                    "error": "An unexpected error occurred.",
                    "code": "unexpected",
                },
                status=500,
            )

    def _serve_file(self, path, cache_control=None):
        if not path.exists() or not path.is_file():
            return self._not_found()

        try:
            body = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            if cache_control:
                self.send_header("Cache-Control", cache_control)
            self.end_headers()
            self.wfile.write(body)
        except CONNECTION_ERRORS:
            self.close_connection = True

    def _not_found(self):
        try:
            self.send_response(404)
            self.end_headers()
        except CONNECTION_ERRORS:
            self.close_connection = True

    def _examples_payload(self):
        payload = []
        for example in EXAMPLES:
            payload.append(
                {
                    "id": example["id"],
                    "title": example["title"],
                    "files": [
                        f"/media/{path.relative_to(PROJECT_DIR).as_posix()}"
                        for path in example_paths(example["id"])
                    ],
                }
            )
        return payload

    def _read_multipart(self, content_length):
        content_type = self.headers.get("Content-Type", "")
        match = re.search(r"boundary=([^;]+)", content_type)
        if not match:
            raise PanoramaError("The form data could not be read.", code="form_invalid")

        boundary = match.group(1).strip('"').encode("utf-8")
        body = self.rfile.read(content_length)
        fields = {}
        files = {}

        for raw_part in body.split(b"--" + boundary):
            part = raw_part.strip()
            if not part or part == b"--":
                continue
            if part.endswith(b"--"):
                part = part[:-2].strip()
            if b"\r\n\r\n" not in part:
                continue

            header_blob, data = part.split(b"\r\n\r\n", 1)
            headers = {}
            for line in header_blob.decode("utf-8", "replace").split("\r\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.strip().lower()] = value.strip()

            disposition = _parse_disposition(headers.get("content-disposition", ""))
            name = disposition.get("name")
            filename = disposition.get("filename")
            data = data.rstrip(b"\r\n")
            if not name:
                continue

            if filename:
                # A field may repeat (the `images` set); the first occurrence
                # stays a plain entry, further ones turn it into a list.
                entry = {"filename": filename, "content": data}
                existing = files.get(name)
                if existing is None:
                    files[name] = entry
                elif isinstance(existing, list):
                    existing.append(entry)
                else:
                    files[name] = [existing, entry]
            else:
                fields[name] = data.decode("utf-8", "replace")

        return fields, files

    def _save_uploads(self, files, upload_dir):
        """Store the uploaded `images` set on disk and return their paths.

        The field may repeat any number of times and the order it arrives in
        does not matter: the pipeline recovers the arrangement itself.
        """
        uploads = [item for item in _upload_list(files, "images") if item["content"]]
        if not uploads:
            raise PanoramaError("Please upload the photos to stitch.", code="upload_missing")
        if len(uploads) < MIN_IMAGES:
            raise PanoramaError(
                f"A panorama needs at least {MIN_IMAGES} photos; {len(uploads)} were uploaded.",
                code="too_few_images",
            )
        if len(uploads) > MAX_IMAGES:
            raise PanoramaError(
                f"At most {MAX_IMAGES} photos can be stitched at once; "
                f"{len(uploads)} were uploaded.",
                code="too_many_images",
            )

        paths = []
        for index, upload in enumerate(uploads, start=1):
            suffix = Path(upload["filename"]).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                suffix = ".jpg"
            destination = upload_dir / f"image_{index}{suffix}"
            destination.write_bytes(upload["content"])

            if cv2.imread(str(destination)) is None:
                try:
                    destination.unlink()
                except OSError:
                    pass
                raise PanoramaError(
                    f"Uploaded file {index} could not be read as an image.",
                    code="upload_not_image",
                    details={"image": index},
                )
            paths.append(destination)
        return paths

    def log_message(self, format, *args):
        print(f"[web] {self.address_string()} - {format % args}")


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # A client that aborts mid-request (RST) can trip an exception in
        # socketserver's own request-reading loop, before our handler code
        # ever runs (e.g. while reading the next request line). That is a
        # normal client disconnect, not a bug, so keep it out of the log.
        exc = sys.exc_info()[1]
        if isinstance(exc, CONNECTION_ERRORS):
            return
        super().handle_error(request, client_address)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    _sweep_job_dirs()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = QuietThreadingHTTPServer(("127.0.0.1", port), PanoramaHandler)
    print(f"Panorama web UI ready: http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
