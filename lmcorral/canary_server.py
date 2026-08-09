"""Local HTTP sink for SSRF probes.

LMCorral judges attempted tool calls from the model stream. If your agent
runtime actually executes those calls, this server records the hits so you can
confirm egress reached the host you bound it to.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class _Handler(BaseHTTPRequestHandler):
    """Log every request path and method; respond with a minimal 200."""

    server: _CanaryHTTPServer  # set by ThreadingHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        """Silence the default stderr access log."""

    def _record(self) -> None:
        """Append this request to the shared hit list."""
        self.server.hits.append(
            {
                "method": self.command,
                "path": self.path,
                "client": self.client_address[0],
            }
        )

    def do_GET(self) -> None:
        """Accept GET and record it."""
        self._record()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self) -> None:
        """Accept POST and record it."""
        length = int(self.headers.get("Content-Length", 0))
        if length:
            _ = self.rfile.read(length)
        self._record()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")


class _CanaryHTTPServer(ThreadingHTTPServer):
    """HTTP server that accumulates request metadata."""

    def __init__(self, address: tuple[str, int]) -> None:
        """Create the server and an empty hit list."""
        super().__init__(address, _Handler)
        self.hits: list[dict[str, str]] = []


class CanaryServer:
    """Start and stop a background canary HTTP listener."""

    def __init__(self) -> None:
        """Prepare an idle server handle."""
        self._httpd: _CanaryHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._base_url = ""

    @property
    def base_url(self) -> str:
        """Root URL of the running server, or empty if not started."""
        return self._base_url

    def start(self, host: str, port: int, path: str = "/canary/ssrf") -> str:
        """Bind `host:port` and serve until `stop()`. Returns the canary URL."""
        if self._httpd is not None:
            return self._base_url
        bind = host if host not in ("0.0.0.0", "") else "127.0.0.1"
        self._httpd = _CanaryHTTPServer((host, port))
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        self._base_url = f"http://{bind}:{port}{path}"
        return self._base_url

    def hits(self) -> list[dict[str, str]]:
        """Requests received since `start`, if the server is running."""
        if self._httpd is None:
            return []
        return list(self._httpd.hits)

    def stop(self) -> None:
        """Shut down the listener."""
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
        self._thread = None
        self._base_url = ""
