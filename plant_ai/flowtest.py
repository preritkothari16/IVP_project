"""
Step 2: run the full user flow 5x in a row in one browser session, with no restarts.
Watches for state leak, stale predictions, stuck UI, and memory growth.
"""

import paths  # central path configuration; see paths.py
import csv
import statistics
from pathlib import Path

from playwright.sync_api import sync_playwright


PROJECT = paths.PROJECT
DATASET = paths.DATASET
SHOTS = PROJECT / "screenshots" / "flowtest"
SHOTS.mkdir(parents=True, exist_ok=True)
URL = "http://127.0.0.1:5173/"

TEST = [r for r in csv.DictReader((DATASET / "manifest.csv").open(encoding="utf-8"))
        if r["split"] == "test" and r["is_dup_copy"] == "False"]
BY = {}
for r in TEST:
    BY.setdefault(r["final_class"], []).append(DATASET / r["path"])

# alternate confident classes so a stale result would be obvious
CYCLE = [
    BY["gray_mold"][0],
    BY["healthy"][0],
    BY["leaf_scorch"][0],
    BY["angular_leafspot"][0],
    BY["powdery_mildew_leaf"][0],
]

LOW_CONF = DATASET / "test" / "leaf_spot" / "leaf_spot_410.jpg"


def visible_text(page):
    return page.inner_text("body")


def main():
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        ctx = b.new_context(viewport={"width": 1280, "height": 900})
        page = ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
                if m.type == "error" else None)

        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(600)

        print(f"{'run':>3s} {'uploaded':34s} {'predicted':22s} {'conf':>7s} {'low':>5s}  {'ms':>6s}  checks")
        print("-" * 104)

        results = []
        for i, path in enumerate(CYCLE, 1):
            # clear any previous state first, exactly as a user would
            btn = page.query_selector("text=Choose a different photo")
            if btn:
                btn.click()
                page.wait_for_timeout(250)

            expect = path.parent.name
            t0 = __import__("time").perf_counter()
            page.set_input_files("input[type=file]:not([capture])", str(path))

            # wait for THIS run's result. innerText is uppercased by CSS, so match
            # case-insensitively on the distinctive result markers.
            try:
                page.wait_for_function(
                    "() => { const t = document.body.innerText.toLowerCase();"
                    " return !t.includes('analysing') &&"
                    " (t.includes('top matches') || t.includes('not confident about this one')); }",
                    timeout=45000,
                )
            except Exception:
                print(f"{i:3d} {path.name:34s} TIMEOUT - result never rendered")
                page.screenshot(path=str(SHOTS / f"run{i}_timeout.png"), full_page=True)
                continue
            ms = (__import__("time").perf_counter() - t0) * 1000
            page.wait_for_timeout(900)  # let animations settle

            body = visible_text(page)
            low = "Not confident about this one" in body

            pred = conf = None
            for line in body.splitlines():
                s = line.strip()
                if s.startswith("Model class:"):
                    pred = s.split(":", 1)[1].strip()
            # confidence is the standalone percentage on its own line under "CONFIDENCE"
            lines = [l.strip() for l in body.splitlines()]
            for idx, l in enumerate(lines):
                if l.upper() == "CONFIDENCE" and idx + 1 < len(lines):
                    for tok in lines[idx + 1].split():
                        if tok.endswith("%"):
                            conf = float(tok.rstrip("%"))
                            break
                    break
            if pred is None:
                # low-confidence card: top-1 is the largest % in the "leaned towards" list,
                # but that list excludes top_class; read the biggest % on the page instead.
                pcts = [float(t.rstrip("%")) for l in lines for t in l.split() if t.endswith("%") and t.rstrip("%").replace(".", "").isdigit()]
                if pcts:
                    conf = max(pcts)

            # checks
            checks = []
            stale = pred is not None and pred != expect and i > 1
            checks.append("STALE!" if stale else "fresh")
            checks.append("has-confidence" if conf is not None else "NO-CONF")
            if "Analysing" in body:
                checks.append("STUCK-LOADING")
            if body.count("Symptoms") > 1:
                checks.append("DUP-CARD")
            if body.count("Not confident about this one") > 1:
                checks.append("DUP-LOW")

            results.append((i, path.name, pred, conf, ms, checks))
            print(f"{i:3d} {path.name[:34]:34s} {str(pred)[:22]:22s} "
                  f"{conf if conf is not None else float('nan'):7.4f} {str(low):>5s}  {ms:6.0f}  {' '.join(checks)}")
            page.screenshot(path=str(SHOTS / f"run{i}.png"), full_page=True)

        # now the low-confidence image twice in a row, to be sure it does not stick
        print("\nlow-confidence image x2 (state-leak check):")
        for i in (6, 7):
            btn = page.query_selector("text=Choose a different photo")
            if btn:
                btn.click()
                page.wait_for_timeout(250)
            page.set_input_files("input[type=file]:not([capture])", str(LOW_CONF))
            try:
                page.wait_for_selector("text=Not confident about this one", timeout=45000)
                page.wait_for_timeout(700)
                body = visible_text(page)
                ok = "Top matches" not in body and "Not confident about this one" in body
                print(f"  run {i}: low-confidence state shown = {ok}  "
                      f"(no result card leaked: {'Top matches' not in body})")
            except Exception:
                print(f"  run {i}: TIMEOUT")

        times = [r[4] for r in results if r[4]]
        if times:
            print(f"\nlatency ms: min={min(times):.0f} median={statistics.median(times):.0f} max={max(times):.0f}")
        else:
            print("\nlatency: no successful runs to measure")

        js = page.evaluate(
            "() => ({heap: performance.memory ? performance.memory.usedJSHeapSize/1048576 : null,"
            " nodes: document.getElementsByTagName('*').length})"
        )
        print(f"JS heap: {js['heap']:.1f} MB | DOM nodes: {js['nodes']}")
        print(f"page errors: {errors if errors else 'NONE'}")

        ctx.close()
        b.close()


if __name__ == "__main__":
    main()
