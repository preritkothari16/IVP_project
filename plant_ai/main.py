"""Single entry point for the strawberry disease classifier.

Subcommands
-----------
    check       verify the environment and that every required input is present (no model run)
    evaluate    score a checkpoint on the TEST split and write metrics + confusion matrix
    serve       start the FastAPI backend and the Vite dev server
    verify      score every TEST image through the live API and compare with the offline eval
    walkthrough drive the web UI in a headless browser and capture screenshots
    all         check -> evaluate -> serve-ready summary

Examples
--------
    python main.py check
    python main.py evaluate                       # uses the shipped checkpoint
    python main.py evaluate --weights runs/.../best.pt --tag mytag
    python main.py serve
    python main.py verify
    python main.py walkthrough

Every path is resolved through paths.py, so the working directory does not matter and no
machine-specific absolute path is baked in.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import paths

PY = sys.executable


def _banner(text):
    print()
    print("=" * 74)
    print(text)
    print("=" * 74)


def _run(cmd, **kw):
    """Run a subprocess, echoing the command, and exit non-zero if it fails."""
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}\n")
    return subprocess.call([str(c) for c in cmd], **kw)


# --------------------------------------------------------------------------- check
def cmd_check(_args):
    _banner("ENVIRONMENT")
    missing = paths.check_environment()

    _banner("DATASET")
    if paths.MANIFEST.exists():
        import csv
        from collections import Counter
        rows = list(csv.DictReader(paths.MANIFEST.open(encoding="utf-8")))
        for split in ("train", "val", "test"):
            c = Counter(r["final_class"] for r in rows
                        if r["split"] == split and r["is_dup_copy"] != "True")
            if c:
                print(f"  {split:6s} {sum(c.values()):5d} images across {len(c)} classes")
    else:
        print(f"  manifest.csv missing at {paths.MANIFEST}")
        print("  build it with:  python prepare_final_dataset.py")

    _banner("SHIPPED MODEL")
    w = paths.SHIPPED_WEIGHTS
    if w.exists():
        print(f"  checkpoint : {w}")
        print(f"  size       : {w.stat().st_size / 1048576:.2f} MB")
        print(f"  deployed   : {paths.BACKEND_WEIGHTS}")
        same = (w.read_bytes() == paths.BACKEND_WEIGHTS.read_bytes()) \
            if paths.BACKEND_WEIGHTS.exists() else False
        print(f"  deployed matches checkpoint: {same}")
    else:
        print(f"  [missing] shipped checkpoint {w}")
        print("  train it with the command in README section 3")

    if missing:
        _banner("RESULT: FAILED")
        print("Missing required inputs:")
        for m in missing:
            print(f"  - {m}")
        return 1
    _banner("RESULT: OK")
    print("Environment looks runnable. Next:  python main.py evaluate")
    return 0


# ------------------------------------------------------------------------ evaluate
def cmd_evaluate(args):
    weights = Path(args.weights) if args.weights else paths.SHIPPED_WEIGHTS
    if not weights.is_absolute():
        weights = (Path.cwd() / weights).resolve()
    if not weights.exists():
        print(f"[error] checkpoint not found: {weights}")
        print("        pass --weights, or train one (README section 3)")
        return 1
    tag = args.tag or paths.SHIPPED_EVAL_TAG
    cmd = [PY, Path(__file__).parent / "evaluate.py", weights, tag]
    if args.tta:
        cmd += ["--tta", str(args.tta)]
    return _run(cmd, cwd=str(paths.PROJECT))


# --------------------------------------------------------------------------- serve
def cmd_serve(_args):
    return _run([PY, Path(__file__).parent / "start_servers.py"], cwd=str(paths.PROJECT))


# --------------------------------------------------------------------------- verify
def cmd_verify(args):
    """Score all TEST images through the live API and cross-check the offline evaluation."""
    import csv
    import uuid
    from concurrent.futures import ThreadPoolExecutor

    import cv2  # noqa: F401  (not needed here, but confirms the env has opencv)

    _banner("LIVE API VERIFICATION")
    try:
        health = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=10).read())
    except Exception as e:  # noqa: BLE001
        print(f"[error] backend not reachable: {e}")
        print("        start it first:  python main.py serve")
        return 1
    print(f"  model_ready : {health.get('model_ready')}")
    print(f"  classes     : {health.get('num_classes')}")

    rows = [r for r in csv.DictReader(paths.MANIFEST.open(encoding="utf-8"))
            if r["split"] == "test" and r["is_dup_copy"] == "False"]
    if not rows:
        print("[error] no TEST rows in manifest")
        return 1

    def probe(r):
        p = paths.DATASET / r["path"]
        b = f"----{uuid.uuid4().hex}"
        body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; "
                f"filename=\"{p.name}\"\r\nContent-Type: image/jpeg\r\n\r\n").encode() + \
            p.read_bytes() + f"\r\n--{b}--\r\n".encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8000/predict", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={b}"})
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    d = json.loads(resp.read())
                return r["final_class"], d["top_class"], d["confidence"]
            except Exception:  # noqa: BLE001
                time.sleep(1)
        return r["final_class"], "ERROR", 0.0

    print(f"\n  scoring {len(rows)} TEST images through /predict ...")
    out = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for t, p_, c in ex.map(probe, rows):
            out.append((t, p_, c))

    ok = sum(1 for t, p_, _ in out if t == p_)
    classes = sorted({t for t, _, _ in out})
    macro_rec = sum(sum(1 for t2, p2, _ in out if t2 == c and t2 == p2)
                    / max(1, sum(1 for t2, _, _ in out if t2 == c)) for c in classes) / len(classes)
    print(f"\n  overall top-1 (accuracy) : {ok / len(out):.4f}")
    print(f"  macro recall             : {macro_rec:.4f}")
    errs = [(t, p_, c) for t, p_, c in out if t != p_]
    print(f"  errors                   : {len(errs)}")
    for t, p_, c in sorted(errs, key=lambda x: -x[2])[:12]:
        print(f"    {t:24s} -> {p_:24s} conf={c:.3f}")

    # cross-check against the offline evaluation if it exists
    ref = paths.EVAL_DIR / args.tag / "predictions.csv" if args.tag else \
        paths.EVAL_DIR / paths.SHIPPED_EVAL_TAG / "predictions.csv"
    if ref.exists():
        d = json.loads((ref.parent / "eval.json").read_text(encoding="utf-8"))
        print(f"\n  offline eval  : macro-F1 {d.get('macro_f1', float('nan')):.4f}  "
              f"macro recall {d.get('macro_recall', d.get('macro_top1', float('nan'))):.4f}")
        print(f"  reference    : {ref}")
    return 0


# ----------------------------------------------------------------------- walkthrough
def cmd_walkthrough(_args):
    return _run([PY, Path(__file__).parent / "walkthrough.py"], cwd=str(paths.PROJECT))


# ------------------------------------------------------------------------------- all
def cmd_all(args):
    rc = cmd_check(args)
    if rc != 0:
        return rc
    rc = cmd_evaluate(args)
    if rc != 0:
        return rc
    _banner("NEXT STEPS")
    print("  python main.py serve        # start backend + frontend")
    print("  python main.py verify       # confirm the served model matches the eval")
    print("  python main.py walkthrough  # capture UI screenshots")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="verify environment and inputs").set_defaults(fn=cmd_check)

    e = sub.add_parser("evaluate", help="score a checkpoint on the TEST split")
    e.add_argument("--weights", help="path to best.pt (default: the shipped checkpoint)")
    e.add_argument("--tag", help="output subdirectory under eval/")
    e.add_argument("--tta", type=int, default=0, help="test-time augmentation views (0=off)")
    e.set_defaults(fn=cmd_evaluate)

    sub.add_parser("serve", help="start backend + frontend").set_defaults(fn=cmd_serve)

    v = sub.add_parser("verify", help="score TEST through the live API and cross-check")
    v.add_argument("--tag", help="eval tag to compare against")
    v.set_defaults(fn=cmd_verify)

    sub.add_parser("walkthrough", help="drive the UI headlessly and screenshot").set_defaults(
        fn=cmd_walkthrough)

    a = sub.add_parser("all", help="check then evaluate")
    a.add_argument("--weights")
    a.add_argument("--tag")
    a.add_argument("--tta", type=int, default=0)
    a.set_defaults(fn=cmd_all)

    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()