"""End-to-end smoke test of the running fullstack app through the vite proxy.

Exercises the real contract: multipart file upload to /api/predict, reading top_class.
Also checks /api/classes and /api/health, and the CORS-free same-origin proxy path.
"""
import io
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path

PROXY = "http://127.0.0.1:5173/api"
TEST = Path(r"D:\ivp\plant_ai\dataset\test")


def get(path):
    with urllib.request.urlopen(f"{PROXY}{path}", timeout=20) as r:
        return r.status, json.loads(r.read())


def post_image(path: Path):
    """multipart/form-data with a single 'file' field, built by hand (no requests dep)."""
    data = path.read_bytes()
    ctype = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    b = "----strawberrytest7f3a"
    body = b"".join([
        f"--{b}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode(),
        f"Content-Type: {ctype}\r\n\r\n".encode(),
        data,
        f"\r\n--{b}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        f"{PROXY}/predict", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={b}"},
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.status, json.loads(r.read())


fails = []

print("=" * 74)
print("1. HEALTH")
print("=" * 74)
try:
    s, j = get("/health")
    print(f"  {s} model_ready={j['model_ready']} classes={j['num_classes']} "
          f"weights_present={j['weights_present']}")
    if not j.get("model_ready"):
        fails.append("health reports model not ready")
except Exception as e:
    fails.append(f"health: {e}")
    print(f"  ERROR {e}")

print()
print("=" * 74)
print("2. CLASSES")
print("=" * 74)
try:
    s, j = get("/classes")
    print(f"  {s} count={j['count']}")
    for c in j["classes"]:
        print(f"      {c}")
except Exception as e:
    fails.append(f"classes: {e}")
    print(f"  ERROR {e}")

print()
print("=" * 74)
print("3. PREDICT - one image per class through the proxy")
print("=" * 74)
ok = wrong = 0
rows = []
for cls in sorted(p.name for p in TEST.iterdir() if p.is_dir()):
    imgs = sorted((TEST / cls).glob("*.jpg"))
    if not imgs:
        continue
    f = imgs[0]
    try:
        s, j = post_image(f)
        tc = j.get("top_class")
        hit = tc == cls
        ok += hit
        wrong += (not hit)
        rows.append((cls, tc, j.get("confidence"), j.get("low_confidence"), hit))
        flag = "OK  " if hit else "MISS"
        print(f"  [{flag}] {cls:24s} -> {tc:24s} conf={j.get('confidence'):.4f}"
              + ("  [low-conf]" if j.get("low_confidence") else ""))
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:160]
        print(f"  [HTTP {e.code}] {cls:24s} {body}")
        fails.append(f"{cls}: HTTP {e.code} {body}")
    except Exception as e:
        print(f"  [ERR] {cls:24s} {e}")
        fails.append(f"{cls}: {e}")

print()
print("=" * 74)
print(f"RESULT: {ok} correct, {wrong} wrong of {ok+wrong}")
print("=" * 74)

if rows:
    lowc = [r for r in rows if r[3]]
    print(f"low-confidence responses: {len(lowc)}"
          + (f"  -> {', '.join(r[0] for r in lowc)}" if lowc else ""))
    keyed = [r for r in rows if r[0] in ("anthracnose_fruit_rot", "powdery_mildew_fruit")]
    if keyed:
        print("known-hard classes: " + ", ".join(
            f"{r[0]}->{r[1]}@{r[2]:.3f}" for r in keyed))

if fails:
    print("\nFAILURES:")
    for f in fails:
        print("  -", f)
else:
    print("\nNo failures.")
