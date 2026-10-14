"""Serve a project folder on 127.0.0.1 and save review notes.

POST /save            writes qc/review.json (key shots)
POST /save-listen     writes qc/listen/review.json (key tracks)
POST /save-decisions  merges qc/decisions.json and copies the previous file to qc/archive/

No picture jobs. No external network.
"""
import json
import re
import shutil
from datetime import datetime
import os
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8765
ROUTES = {
    "/save": (ROOT / "qc" / "review.json", "shots"),
    "/save-listen": (ROOT / "qc" / "listen" / "review.json", "tracks"),
}
ITEM_ID = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
REL_PATH = re.compile(r"^[A-Za-z0-9_./-]{1,240}$")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _json(self, code, obj):
        body = (json.dumps(obj) + "\n").encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", "0"))
        if n > 2_000_000:
            self.send_error(413)
            return None
        try:
            data = json.loads(self.rfile.read(n).decode())
        except json.JSONDecodeError:
            self.send_error(400)
            return None
        if not isinstance(data, dict):
            self.send_error(400)
            return None
        return data

    def _decisions(self, data):
        items = data.get("items")
        if not isinstance(items, dict) or not items or len(items) > 40:
            self.send_error(400)
            return
        clean = {}
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        for key, item in items.items():
            if not isinstance(key, str) or not ITEM_ID.match(key) or not isinstance(item, dict):
                self.send_error(400)
                return
            choice = item.get("choice")
            note = item.get("note", "")
            evidence = item.get("evidence", [])
            if not isinstance(choice, str) or not choice.strip() or len(choice) > 80:
                self.send_error(400)
                return
            if not isinstance(note, str) or len(note) > 4000:
                self.send_error(400)
                return
            if not isinstance(evidence, list) or len(evidence) > 20:
                self.send_error(400)
                return
            paths = []
            for path in evidence:
                if not isinstance(path, str) or not REL_PATH.match(path) or path.startswith("/") or ".." in path:
                    self.send_error(400)
                    return
                paths.append(path)
            clean[key] = {"choice": choice.strip(), "note": note, "evidence": paths, "at": now}
        dest = ROOT / "qc" / "decisions.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        archive = ROOT / "qc" / "archive"
        archive.mkdir(parents=True, exist_ok=True)
        prev = {"items": {}}
        if dest.exists():
            shutil.copy2(dest, archive / f"decisions-{datetime.now().strftime('%Y%m%dT%H%M%S')}.json")
            try:
                prev = json.loads(dest.read_text())
            except json.JSONDecodeError:
                prev = {"items": {}}
        merged = dict(prev.get("items") or {})
        merged.update(clean)
        out = {"saved": now, "items": merged}
        dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
        self._json(200, {"ok": True, "saved": now, "ids": sorted(clean)})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        data = self._read_json()
        if data is None:
            return
        if path == "/save-decisions":
            self._decisions(data)
            return
        route = ROUTES.get(path)
        if route is None:
            self.send_error(404)
            return
        dest, key = route
        if not isinstance(data.get(key), list):
            self.send_error(400)
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        self._json(200, {"ok": True})

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    ThreadingHTTPServer((os.environ.get("REVIEW_BIND", "127.0.0.1"), PORT), Handler).serve_forever()
