from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
import json
import mimetypes
import re
import sys
import uuid

from panorama_pipeline import EXAMPLES, PROJECT_DIR, PanoramaError, example_paths, stitch_pair


STATIC_DIR = PROJECT_DIR / "static"
OUTPUT_DIR = PROJECT_DIR / "web_outputs"
UPLOAD_DIR = PROJECT_DIR / "web_uploads"


def _safe_join(root, relative_path):
    path = (root / relative_path).resolve()
    if not str(path).startswith(str(root.resolve())):
        raise ValueError("Invalid path")
    return path


def _json_response(handler, payload, status=200):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _parse_disposition(header_value):
    values = {}
    for key, value in re.findall(r'(\w+)="([^"]*)"', header_value):
        values[key] = value
    return values


class PanoramaHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/":
            return self._serve_file(PROJECT_DIR / "index.html")

        if path == "/api/examples":
            return _json_response(self, {"examples": self._examples_payload()})

        if path.startswith("/static/"):
            try:
                return self._serve_file(_safe_join(STATIC_DIR, path.removeprefix("/static/")))
            except ValueError:
                return self._not_found()

        if path.startswith("/media/"):
            try:
                return self._serve_file(_safe_join(PROJECT_DIR, path.removeprefix("/media/")))
            except ValueError:
                return self._not_found()

        if path.startswith("/generated/"):
            try:
                return self._serve_file(_safe_join(OUTPUT_DIR, path.removeprefix("/generated/")))
            except ValueError:
                return self._not_found()

        return self._not_found()

    def do_POST(self):
        if urlparse(self.path).path != "/api/stitch":
            return self._not_found()

        try:
            fields, files = self._read_multipart()
            job_id = uuid.uuid4().hex[:12]
            job_dir = OUTPUT_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            mode = fields.get("mode", "example")
            if mode == "example":
                left_path, right_path = example_paths(fields.get("example", "clock"))
                source = {
                    "left": f"/media/{left_path.relative_to(PROJECT_DIR).as_posix()}",
                    "right": f"/media/{right_path.relative_to(PROJECT_DIR).as_posix()}",
                }
            else:
                upload_dir = UPLOAD_DIR / job_id
                upload_dir.mkdir(parents=True, exist_ok=True)
                left_path = self._save_upload(files, "leftImage", upload_dir / "sol.jpg")
                right_path = self._save_upload(files, "rightImage", upload_dir / "sag.jpg")
                source = {
                    "left": f"/media/{left_path.relative_to(PROJECT_DIR).as_posix()}",
                    "right": f"/media/{right_path.relative_to(PROJECT_DIR).as_posix()}",
                }

            result = stitch_pair(left_path, right_path, job_dir)
            files_payload = {
                name: f"/generated/{job_id}/{filename}"
                for name, filename in result["files"].items()
            }
            return _json_response(
                self,
                {
                    "ok": True,
                    "jobId": job_id,
                    "source": source,
                    "metrics": result["metrics"],
                    "files": files_payload,
                },
            )
        except PanoramaError as exc:
            return _json_response(self, {"ok": False, "error": str(exc)}, status=422)
        except Exception as exc:
            return _json_response(self, {"ok": False, "error": f"Beklenmeyen hata: {exc}"}, status=500)

    def _serve_file(self, path):
        if not path.exists() or not path.is_file():
            return self._not_found()

        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self):
        self.send_response(404)
        self.end_headers()

    def _examples_payload(self):
        payload = []
        for example in EXAMPLES:
            left_path, right_path = example_paths(example["id"])
            payload.append(
                {
                    "id": example["id"],
                    "title": example["title"],
                    "left": f"/media/{left_path.relative_to(PROJECT_DIR).as_posix()}",
                    "right": f"/media/{right_path.relative_to(PROJECT_DIR).as_posix()}",
                }
            )
        return payload

    def _read_multipart(self):
        content_type = self.headers.get("Content-Type", "")
        match = re.search(r"boundary=([^;]+)", content_type)
        if not match:
            raise PanoramaError("Form verisi okunamadi.")

        boundary = match.group(1).strip('"').encode("utf-8")
        content_length = int(self.headers.get("Content-Length", "0"))
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
                files[name] = {"filename": filename, "content": data}
            else:
                fields[name] = data.decode("utf-8", "replace")

        return fields, files

    def _save_upload(self, files, field_name, destination):
        upload = files.get(field_name)
        if not upload or not upload["content"]:
            raise PanoramaError("Lutfen sol ve sag gorseli yukleyin.")

        suffix = Path(upload["filename"]).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            suffix = ".jpg"
        destination = destination.with_suffix(suffix)
        destination.write_bytes(upload["content"])
        return destination

    def log_message(self, format, *args):
        print(f"[web] {self.address_string()} - {format % args}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), PanoramaHandler)
    print(f"Panorama arayuzu hazir: http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
