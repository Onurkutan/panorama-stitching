import http.client
import json
import threading
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import pytest

import app


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
    assert len(payload["examples"]) == 3


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
        resp.read()
    finally:
        conn.close()


def test_stitch_example_mode(live_server, multipart_body):
    body, content_type = multipart_body({"mode": "example", "example": "clock"}, {})
    request = urllib.request.Request(
        f"{live_server}/api/stitch", data=body, headers={"Content-Type": content_type}
    )
    with urllib.request.urlopen(request) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read())

    assert payload["ok"] is True
    panorama_url = payload["files"]["panorama"]

    with urllib.request.urlopen(f"{live_server}{panorama_url}") as file_resp:
        assert file_resp.status == 200
        assert file_resp.headers.get("Cache-Control") == "no-store"


def test_stitch_upload_non_image_rejected(live_server, multipart_body):
    body, content_type = multipart_body(
        {"mode": "upload"},
        {
            "leftImage": ("left.txt", b"not an image"),
            "rightImage": ("right.txt", b"not an image either"),
        },
    )
    request = urllib.request.Request(
        f"{live_server}/api/stitch", data=body, headers={"Content-Type": content_type}
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(request)
    assert exc_info.value.code == 422
