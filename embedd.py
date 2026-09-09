#!/usr/bin/env python3
"""embedd - one always-on semantic index over a folder of markdown, on CPU.

Two processes make this work:

  llama-server --embedding   the model, CPU only, localhost         (EMBED_URL)
  embedd.py                  the index + search API, on the network (PORT)

Everything that wants retrieval - a voice assistant, an autonomous agent, a
CLI - asks this one index instead of building its own.  One model, one
chunking scheme, so every caller sees the same neighbourhood for a query.

API (JSON in, JSON out):

  GET  /health                      model, dim, chunk/file counts, index age
  POST /search  {q, k, path}        top-k chunks; `path` is an optional prefix filter
  POST /embed   {input: [str,...]}  raw vectors, for callers with their own store
  POST /reindex {full: false}       force an incremental (or full) pass now

The index refreshes itself on a timer, so callers never have to think about it.

Env: EMBED_VAULT EMBED_STATE EMBED_URL EMBED_PORT EMBED_HOST EMBED_INTERVAL
     EMBED_EXCLUDE EMBED_MODEL
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

VAULT    = Path(os.environ.get("EMBED_VAULT", "/data/vault"))
STATE    = Path(os.environ.get("EMBED_STATE", os.path.expanduser("~/.local/state/vault-embed")))
EMBED_URL= os.environ.get("EMBED_URL", "http://127.0.0.1:11476")
MODEL    = os.environ.get("EMBED_MODEL", "nomic-embed-text-v1.5")
HOST     = os.environ.get("EMBED_HOST", "0.0.0.0")
PORT     = int(os.environ.get("EMBED_PORT", "11475"))
INTERVAL = int(os.environ.get("EMBED_INTERVAL", "300"))

# Dot-directories are always skipped: .stversions alone holds ~9k stale copies
# of the same notes, which would swamp every search result.
SKIP_DIRS = {".git", ".stfolder", ".stversions", ".obsidian", ".trash", "node_modules"}
EXCLUDE   = [s for s in os.environ.get("EMBED_EXCLUDE", "junk,Templates,Excalidraw").split(",") if s]

CHUNK_CHARS = 1200      # pack paragraphs up to this, split on headings first
MIN_CHARS   = 40        # a chunk shorter than this carries no meaning
MAX_CHARS   = 4000      # hard cap per chunk sent to the model
BATCH_CHARS = 12000     # per embed request, keeps us inside the server's batch
BATCH_N     = 8

DB = STATE / "index.db"

_lock  = threading.Lock()      # guards _M/_meta swap
_M: np.ndarray | None = None   # (n, dim) unit-normalised
_meta: list = []               # parallel [(path, heading, text)]
_stats = {"chunks": 0, "files": 0, "dim": 0, "last_index": 0.0,
          "last_pass_s": 0.0, "indexing": False, "error": "",
          "to_index": 0, "done": 0}


# ---------------------------------------------------------------- embedding

def embed(texts: list[str], tries: int = 8) -> list[list[float]]:
    """Vectors for `texts`, via the llama.cpp OpenAI-compatible endpoint.

    The model server answers 503 for the first half-minute or so while it loads,
    and systemd may restart it under us, so a call waits rather than failing the
    whole pass."""
    body = json.dumps({"model": MODEL, "input": texts}).encode()
    for attempt in range(tries):
        req = urllib.request.Request(EMBED_URL + "/v1/embeddings", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                out = json.load(r)
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
            code = getattr(e, "code", None)
            if attempt == tries - 1 or (code is not None and code not in (500, 502, 503)):
                raise
            time.sleep(min(2 * (attempt + 1), 15))
    rows = sorted(out["data"], key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in rows]


def embed_batched(texts: list[str]) -> list[list[float]]:
    vecs: list[list[float]] = []
    buf: list[str] = []
    n = 0
    for t in texts:
        if buf and (len(buf) >= BATCH_N or n + len(t) > BATCH_CHARS):
            vecs += embed(buf); buf = []; n = 0
        buf.append(t); n += len(t)
    if buf:
        vecs += embed(buf)
    return vecs


# ---------------------------------------------------------------- chunking

def chunks(text: str) -> list[tuple[str, str]]:
    """Split on headings, then pack paragraphs to ~CHUNK_CHARS."""
    out: list[tuple[str, str]] = []
    for part in re.split(r"\n(?=#{1,6}\s)", text):
        head = part.split("\n", 1)[0].strip()[:120] if part.lstrip().startswith("#") else ""
        buf = ""
        for para in re.split(r"\n\s*\n", part):
            para = para.strip()
            if not para:
                continue
            if len(buf) + len(para) > CHUNK_CHARS and buf:
                out.append((head, buf)); buf = para
            else:
                buf = (buf + "\n\n" + para).strip()
        if buf:
            out.append((head, buf))
    return [(h, c) for h, c in out if len(c) > MIN_CHARS]


def walk() -> list[tuple[str, float]]:
    files = []
    for root, dirs, fs in os.walk(VAULT):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS]
        rel_root = os.path.relpath(root, VAULT)
        top = rel_root.split(os.sep)[0]
        if top in EXCLUDE:
            dirs[:] = []
            continue
        for f in sorted(fs):
            if f.endswith(".md") and not f.startswith("."):
                p = os.path.join(root, f)
                try:
                    files.append((os.path.relpath(p, VAULT), os.path.getmtime(p)))
                except OSError:
                    pass
    return files


# ---------------------------------------------------------------- index

def open_db() -> sqlite3.Connection:
    STATE.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS chunks(
        id INTEGER PRIMARY KEY, path TEXT, heading TEXT, text TEXT, mtime REAL, vec BLOB)""")
    con.execute("CREATE INDEX IF NOT EXISTS ix_path ON chunks(path)")
    return con


def load() -> None:
    """Pull the whole index into RAM as one matrix. 10k chunks x 768 = 30 MB."""
    global _M, _meta
    con = open_db()
    rows = con.execute("SELECT path,heading,text,vec FROM chunks").fetchall()
    con.close()
    if not rows:
        with _lock:
            _M, _meta = None, []
        _stats.update(chunks=0, dim=0)
        return
    M = np.vstack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    with _lock:
        _M = M
        _meta = [(r[0], r[1], r[2]) for r in rows]
    _stats.update(chunks=len(rows), dim=int(M.shape[1]),
                  files=len({r[0] for r in rows}))


def index_pass(full: bool = False) -> dict:
    t0 = time.time()
    _stats["indexing"] = True
    con = open_db()
    if full:
        con.execute("DELETE FROM chunks"); con.commit()
    known = {r[0]: r[1] for r in con.execute("SELECT path,MAX(mtime) FROM chunks GROUP BY path")}
    seen, new, changed = set(), 0, 0
    todo = walk()
    _stats["to_index"] = sum(1 for rel, mt in todo
                             if known.get(rel) is None or abs(known[rel] - mt) >= 1)
    _stats["done"] = 0
    for rel, mt in todo:
        seen.add(rel)
        if known.get(rel) is not None and abs(known[rel] - mt) < 1:
            continue
        con.execute("DELETE FROM chunks WHERE path=?", (rel,))
        try:
            text = (VAULT / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        cs = chunks(text)
        if not cs:
            continue
        changed += 1
        # The path and heading ride along in the embedded text: a note's title
        # is often the only place the subject is named.
        payload = [f"{rel}\n{h}\n{c}"[:MAX_CHARS] for h, c in cs]
        try:
            vecs = embed_batched(payload)
        except (urllib.error.URLError, OSError, KeyError) as e:
            _stats["error"] = f"{type(e).__name__}: {e}"
            con.commit(); con.close(); _stats["indexing"] = False
            raise
        for (h, c), v in zip(cs, vecs):
            con.execute("INSERT INTO chunks(path,heading,text,mtime,vec) VALUES(?,?,?,?,?)",
                        (rel, h, c, mt, np.asarray(v, dtype=np.float32).tobytes()))
            new += 1
        _stats["done"] += 1
        if _stats["done"] % 25 == 0:
            con.commit()
            # Publish as we go: the first pass over a big vault takes hours, and
            # a search that returns nothing the whole time looks broken.
            load()
            log(f"index: {_stats['done']}/{_stats['to_index']} files, {new} chunks")
    # Dot-files were indexed by an earlier version; drop them on sight.
    gone = [p for p in known if p not in seen
            or any(s.startswith(".") for s in p.split(os.sep))]
    for p in gone:
        con.execute("DELETE FROM chunks WHERE path=?", (p,))
    con.commit()
    con.close()
    load()
    _stats.update(indexing=False, last_index=time.time(), last_pass_s=round(time.time() - t0, 1),
                  error="")
    return {"new_chunks": new, "changed_files": changed, "removed_files": len(gone),
            "chunks": _stats["chunks"], "files": _stats["files"],
            "seconds": _stats["last_pass_s"]}


def indexer() -> None:
    while True:
        try:
            r = index_pass()
            if r["changed_files"] or r["removed_files"]:
                log(f"index: +{r['new_chunks']} chunks from {r['changed_files']} file(s), "
                    f"-{r['removed_files']} gone, {r['chunks']} total, {r['seconds']}s")
            time.sleep(INTERVAL)
        except Exception as e:                                    # noqa: BLE001
            log(f"index failed: {type(e).__name__}: {e}")
            time.sleep(30)


# ---------------------------------------------------------------- search

def search(q: str, k: int = 6, path_prefix: str = "") -> list[dict]:
    q = " ".join(str(q).split())
    if not q:
        return []
    with _lock:
        M, meta = _M, _meta
    if M is None:
        return []
    qv = np.asarray(embed([q])[0], dtype=np.float32)
    qv /= (np.linalg.norm(qv) + 1e-9)
    sims = M @ qv
    if path_prefix:
        mask = np.array([m[0].startswith(path_prefix) for m in meta])
        sims = np.where(mask, sims, -1.0)
    k = max(1, min(int(k), len(meta)))
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return [{"score": round(float(sims[i]), 4), "path": meta[i][0],
             "heading": meta[i][1], "text": meta[i][2]}
            for i in top if sims[i] > -1.0]


# ---------------------------------------------------------------- http

def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        if self.path.split("?")[0] in ("/health", "/", "/stats"):
            age = time.time() - _stats["last_index"] if _stats["last_index"] else None
            self._send({**_stats, "model": MODEL, "vault": str(VAULT),
                        "age_s": round(age, 1) if age else None})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        p = self.path.split("?")[0]
        b = self._body()
        try:
            if p == "/search":
                self._send({"hits": search(b.get("q", ""), b.get("k", 6), b.get("path", ""))})
            elif p == "/embed":
                inp = b.get("input") or b.get("texts") or []
                if isinstance(inp, str):
                    inp = [inp]
                self._send({"embeddings": embed_batched([str(t)[:MAX_CHARS] for t in inp])})
            elif p == "/reindex":
                self._send(index_pass(bool(b.get("full"))))
            else:
                self._send({"error": "not found"}, 404)
        except Exception as e:                                    # noqa: BLE001
            self._send({"error": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, *a):
        pass


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "index":
        print(json.dumps(index_pass("--full" in sys.argv), indent=1))
        return 0
    if len(sys.argv) > 2 and sys.argv[1] == "search":
        load()
        for h in search(" ".join(sys.argv[2:])):
            print(f"[{h['score']}] {h['path']}" + (f"  # {h['heading']}" if h["heading"] else ""))
            print("   " + h["text"][:300].replace("\n", " "))
        return 0
    load()
    log(f"loaded {_stats['chunks']} chunks / {_stats['files']} files, dim {_stats['dim']}")
    threading.Thread(target=indexer, name="indexer", daemon=True).start()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    log(f"listening on {HOST}:{PORT}  vault={VAULT}  model={MODEL}")
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
