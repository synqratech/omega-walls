from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading

import pytest

from omega.projector.api_hybrid import normalization as norm


class _RedirectHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        self.send_response(302)
        self.send_header("Location", "https://evil.example/v1/chat/completions")
        self.end_headers()

    def log_message(self, format, *args):  # noqa: A002
        return


class _JsonHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        _ = self.rfile.read(int(self.headers.get("content-length", "0") or "0"))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))

    def log_message(self, format, *args):  # noqa: A002
        return


def _server(handler):
    srv = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    return srv


def test_post_json_blocks_provider_redirects_by_default():
    srv = _server(_RedirectHandler)
    try:
        url = f"http://127.0.0.1:{srv.server_port}/v1/chat/completions"
        with pytest.raises(RuntimeError, match="provider_redirect_blocked"):
            norm.post_json(url=url, payload={"x": 1}, headers={}, timeout_sec=2.0)
    finally:
        srv.shutdown()


def test_post_json_without_redirect_accepts_json():
    srv = _server(_JsonHandler)
    try:
        url = f"http://127.0.0.1:{srv.server_port}/v1/chat/completions"
        out = norm.post_json(url=url, payload={"x": 1}, headers={}, timeout_sec=2.0)
        assert out["ok"] is True
    finally:
        srv.shutdown()
