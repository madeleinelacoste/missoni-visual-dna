"""Serve the project folder locally so tools/reference_builder.html can read the
runway photos, and accept its result at POST /save -> docs/reference.json.

    python tools/serve_reference_builder.py   # then open http://127.0.0.1:8010/tools/reference_builder.html

Binds to 127.0.0.1 only: the photos are never exposed beyond this machine.
"""

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 8010


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/save":
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers["Content-Length"]))
        ref = json.loads(body)
        assert {"looks", "embeddings", "calibration"} <= ref.keys()
        (ROOT / "docs" / "reference.json").write_text(json.dumps(ref, separators=(",", ":")))
        self.send_response(200)
        self.end_headers()
        print(f"saved docs/reference.json ({len(ref['looks'])} looks)")


if __name__ == "__main__":
    print(f"Open http://127.0.0.1:{PORT}/tools/reference_builder.html (takes a few minutes; keep the tab visible)")
    ThreadingHTTPServer(("127.0.0.1", PORT), partial(Handler, directory=str(ROOT))).serve_forever()
