import http.client
import json
import threading
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import pytest

import app
from panorama_stitching.pipeline import MAX_IMAGES


def _upload_body(multipart_body, paths):
    """A multipart upload carrying one `images` field per photo."""
    return multipart_body(
        {"mode": "upload"},
        {"images": [(path.name, path.read_bytes()) for path in paths]},
    )


def _post(live_server, body, content_type):
    request = urllib.request.Request(
        f"{live_server}/api/stitch", data=body, headers={"Content-Type": content_type}
    )
    with urllib.request.urlopen(request) as resp:
        return resp.status, json.loads(resp.read())


def _post_expecting_error(live_server, body, content_type):
    request = urllib.request.Request(
        f"{live_server}/api/stitch", data=body, headers={"Content-Type": content_type}
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(request)
    return exc_info.value.code, json.loads(exc_info.value.read())


@pytest.fixture
def live_server(tmp_path, monkeypatch):
    output_dir = tmp_path / "web_outputs"
    upload_dir = tmp_path / "web_uploads"
    output_dir.mkdir()
    upload_dir.mkdir()

    # Keep everything the tests write out of the real project directories.
    monkeypatch.setattr(app, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(app, "UPLOAD_DIR", upload_dir)
    monkeypatch.setitem(app.MEDIA_ROOTS, "web_outputs", output_dir)
    monkeypatch.setitem(app.MEDIA_ROOTS, "web_uploads", upload_dir)

    server = app.QuietThreadingHTTPServer(("127.0.0.1", 0), app.PanoramaHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_get_index(live_server):
    with urllib.request.urlopen(f"{live_server}/") as resp:
        assert resp.status == 200
        assert resp.headers.get_content_type() == "text/html"


def test_get_examples(live_server):
    with urllib.request.urlopen(f"{live_server}/api/examples") as resp:
        assert resp.status == 200
        payload = json.loads(resp.read())
    assert len(payload["examples"]) == 4
    assert all(len(example["files"]) >= 2 for example in payload["examples"])


def test_media_app_py_not_served(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{live_server}/media/app.py")
    assert exc_info.value.code == 404


def test_media_image_served(live_server):
    with urllib.request.urlopen(f"{live_server}/media/images/Clock/sol1.jpg") as resp:
        assert resp.status == 200
        assert resp.headers.get_content_type() == "image/jpeg"
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"


def test_media_path_traversal_blocked(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{live_server}/media/../app.py")
    assert exc_info.value.code == 404


def test_stitch_content_length_too_large(live_server):
    parts = urlsplit(live_server)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    try:
        conn.putrequest("POST", "/api/stitch")
        conn.putheader("Content-Type", "multipart/form-data; boundary=x")
        conn.putheader("Content-Length", str(app.MAX_UPLOAD_BYTES + 1))
        conn.endheaders()
        resp = conn.getresponse()
        assert resp.status == 413
        payload = json.loads(resp.read())
    finally:
        conn.close()

    assert payload["ok"] is False
    assert payload["code"] == "upload_too_large"


def test_stitch_without_content_length(live_server):
    """No Content-Length at all: a request-level 400 that still carries a code."""
    parts = urlsplit(live_server)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    try:
        conn.putrequest("POST", "/api/stitch", skip_accept_encoding=True)
        conn.putheader("Content-Type", "multipart/form-data; boundary=x")
        conn.endheaders()
        resp = conn.getresponse()
        assert resp.status == 400
        payload = json.loads(resp.read())
    finally:
        conn.close()

    assert payload["ok"] is False
    assert payload["code"] == "form_invalid"


def test_stitch_example_mode(live_server, multipart_body):
    """The bundled examples run through the set pipeline, so their reply is
    shaped exactly like an upload of two photos."""
    body, content_type = multipart_body({"mode": "example", "example": "clock"}, {})
    status, payload = _post(live_server, body, content_type)

    assert status == 200
    assert payload["ok"] is True
    assert payload["metrics"]["imageCount"] == 2
    assert len(payload["source"]) == 2

    files = payload["files"]
    assert files["pairs"] == [[1, 2]]
    assert len(files["keypoints"]) == 2
    assert len(files["matches"]) == len(files["ransac"]) == 1
    assert files["panorama"].startswith(f"/generated/{payload['jobId']}/")

    with urllib.request.urlopen(f"{live_server}{files['panorama']}") as file_resp:
        assert file_resp.status == 200
        assert file_resp.headers.get("Cache-Control") == "no-store"


def test_stitch_upload_image_set(live_server, multipart_body, crop_set_paths):
    """Three photos through the repeated `images` field, in no particular order."""
    body, content_type = _upload_body(multipart_body, list(reversed(crop_set_paths)))
    status, payload = _post(live_server, body, content_type)

    assert status == 200
    assert payload["ok"] is True
    assert payload["metrics"]["imageCount"] == 3
    assert len(payload["metrics"]["keypoints"]) == 3
    assert len(payload["source"]) == 3

    files = payload["files"]
    assert len(files["keypoints"]) == 3
    assert len(files["matches"]) == len(files["ransac"]) == len(files["pairs"]) == 2
    for url in [files["panorama"], *files["keypoints"], *files["matches"], *files["ransac"]]:
        with urllib.request.urlopen(f"{live_server}{url}") as file_resp:
            assert file_resp.status == 200


def test_stitch_upload_single_image_rejected(live_server, multipart_body, crop_set_paths):
    body, content_type = _upload_body(multipart_body, crop_set_paths[:1])
    status, payload = _post_expecting_error(live_server, body, content_type)

    assert status == 422
    assert payload["ok"] is False
    assert payload["code"] == "too_few_images"


def test_stitch_upload_too_many_images_rejected(live_server, multipart_body, crop_set_paths):
    body, content_type = _upload_body(multipart_body, [crop_set_paths[0]] * (MAX_IMAGES + 1))
    status, payload = _post_expecting_error(live_server, body, content_type)

    assert status == 422
    assert payload["code"] == "too_many_images"


def test_stitch_upload_without_images_rejected(live_server, multipart_body):
    body, content_type = multipart_body({"mode": "upload"}, {})
    status, payload = _post_expecting_error(live_server, body, content_type)

    assert status == 422
    assert payload["code"] == "upload_missing"


def test_stitch_upload_non_image_rejected(live_server, multipart_body):
    body, content_type = multipart_body(
        {"mode": "upload"},
        {"images": [("first.txt", b"not an image"), ("second.txt", b"not an image either")]},
    )
    status, payload = _post_expecting_error(live_server, body, content_type)

    assert status == 422
    assert payload["ok"] is False
    assert payload["code"] == "upload_not_image"
    # The reply names which of the uploaded files could not be decoded.
    assert payload["details"] == {"image": 1}


def test_stitch_downscales_large_inputs(live_server, multipart_body, monkeypatch):
    # The clock pair is 1300 px wide; with a 600 px cap the server must
    # scale both inputs down and report the factor it used.
    monkeypatch.setattr(app, "MAX_INPUT_SIDE", 600)
    body, content_type = multipart_body({"mode": "example", "example": "clock"}, {})
    _status, payload = _post(live_server, body, content_type)

    assert payload["ok"] is True
    assert 0 < payload["metrics"]["inputScale"] < 1
    assert payload["metrics"]["panoramaWidth"] < 1300


def test_images_served_site_relative(live_server):
    with urllib.request.urlopen(f"{live_server}/images/Clock/sol1.jpg") as resp:
        assert resp.status == 200
        assert resp.headers.get("Content-Type") == "image/jpeg"


def test_static_examples_match_pipeline():
    # static/app.js carries a copy of EXAMPLES for the server-less (Pyodide) mode.
    import re

    from panorama_stitching import pipeline

    source = (pipeline.PROJECT_DIR / "static" / "app.js").read_text(encoding="utf-8")
    pattern = re.compile(
        r'id: "(?P<id>\w+)", title: "(?P<title>[^"]+)", folder: "(?P<folder>[^"]+)", '
        r"files: \[(?P<files>[^\]]*)\]"
    )
    found = [
        {
            "id": m["id"],
            "title": m["title"],
            "folder": m["folder"],
            "files": re.findall(r'"([^"]+)"', m["files"]),
        }
        for m in pattern.finditer(source)
    ]
    expected = [
        {k: example[k] for k in ("id", "title", "folder", "files")} for example in pipeline.EXAMPLES
    ]
    assert found == expected
