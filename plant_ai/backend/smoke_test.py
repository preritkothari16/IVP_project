"""Smoke-test the FastAPI backend against a live uvicorn server."""

import io
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

from PIL import Image

PORT = 8011
BASE = f"http://127.0.0.1:{PORT}"


def multipart(field, filename, content, ctype="image/jpeg"):
    b = "----boundary123"
    parts = [
        f"--{b}\r\n".encode(),
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode(),
        f"Content-Type: {ctype}\r\n\r\n".encode(),
        content,
        f"\r\n--{b}--\r\n".encode(),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={b}"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=15) as r:
        return r.status, json.load(r)


def post(path, filename, content, ctype="image/jpeg"):
    body, hdr = multipart("file", filename, content, ctype)
    req = urllib.request.Request(BASE + path, data=body, headers={"Content-Type": hdr})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def jpeg(color=(12, 120, 20), size=(96, 96)):
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, "JPEG")
    return b.getvalue()


def main():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=r"D:\ivp\plant_ai\backend",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        time.sleep(15)
        for path in ("/health", "/classes"):
            try:
                st, js = get(path)
                print(f"GET {path} -> {st}")
                print(json.dumps(js, indent=2)[:700])
            except Exception as exc:
                print(f"GET {path} FAILED: {exc}")

        print("\nPOST /predict (expect 503 - no weights yet)")
        print(post("/predict", "t.jpg", jpeg()))
    finally:
        proc.terminate()
        time.sleep(2)
        try:
            out = proc.stdout.read()
        except Exception:
            out = ""
        print("\n--- server log tail ---")
        print("\n".join((out or "").strip().splitlines()[-15:]))


if __name__ == "__main__":
    main()
