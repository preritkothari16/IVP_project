"""
Step 4: the presentable demo walkthrough. Captures each step with a real screenshot and
reports exactly what the UI shows. Runs against the already-running servers.
"""

import paths  # central path configuration; see paths.py
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright


PROJECT = paths.PROJECT
DEMO = paths.DEMO_IMAGES
OUT = paths.SCREENSHOTS / "walkthrough"
APP_URL = "http://127.0.0.1:5173"
API_URL = "http://127.0.0.1:8000/health"


def preflight():
    """Fail with an actionable message instead of a Playwright stack trace when the servers
    are not running. Run `python start_servers.py` first."""
    problems = []
    for name, url in (("frontend", APP_URL), ("backend", API_URL)):
        try:
            urllib.request.urlopen(url, timeout=5).read(1)
            print(f"  [ok] {name:9s} {url}")
        except Exception as e:  # noqa: BLE001 - want the reason, whatever it is
            problems.append(f"{name} not reachable at {url} ({e})")
            print(f"  [--] {name:9s} {url}  NOT REACHABLE")
    if problems:
        raise SystemExit(
            "\nwalkthrough.py needs both servers running. Start them first:\n\n"
            "    python start_servers.py\n\n"
            "then re-run:\n\n    python walkthrough.py\n\n"
            "Details:\n  " + "\n  ".join(problems)
        )
OUT.mkdir(parents=True, exist_ok=True)
URL = APP_URL

CONFIDENT = DEMO / "gray_mold__gray_mold_49.jpg"
LOW_CONF = DEMO / "HARD__low_confidence__leaf_spot_410.jpg"
AMBIG = DEMO / "HARD__ambiguous__powdery_mildew_fruit_123.jpg"
PM_LEAF = DEMO / "powdery_mildew_leaf__powdery_mildew_leaf_201.jpg"


def wait_result(page):
    page.wait_for_function(
        "() => { const t=document.body.innerText.toLowerCase();"
        " return !t.includes('analysing') && (t.includes('top matches')||t.includes('not confident about this one')); }",
        timeout=60000,
    )
    page.wait_for_timeout(1000)


def main():
    preflight()

    with sync_playwright() as pw:
        b = pw.chromium.launch()

        # ============ STEP 1: landing ============
        ctx = b.new_context(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(900)
        page.screenshot(path=str(OUT / "step1_landing.png"), full_page=True)
        body = page.inner_text("body")
        print("STEP 1 landing")
        print("  title        :", page.title())
        print("  has dropzone :", "Drag a photo here" in body)
        print("  has camera   :", "Use camera" in body)
        print("  has detects  :", "Detects:" in body)
        print("  has limits   :", "Known limitations of this model" in body)
        print("  footer disc  :", "not a substitute" in body)

        # ============ STEP 2: confident result ============
        page.set_input_files("input[type=file]:not([capture])", str(CONFIDENT))
        wait_result(page)
        page.screenshot(path=str(OUT / "step2_confident_result.png"), full_page=True)
        body = page.inner_text("body")
        print("\nSTEP 2 confident result (gray_mold__gray_mold_49.jpg)")
        print("  disease name :", "Grey mould" in body)
        print("  severity     :", "High severity" in body)
        print("  confidence   :", [l for l in body.splitlines() if l.strip().endswith("%")][:1])
        print("  top-3 shown  :", "TOP MATCHES" in body.upper())
        print("  4 tabs       :", page.get_by_role("tab").count() == 4,
              [page.get_by_role("tab").nth(i).inner_text() for i in range(page.get_by_role("tab").count())])
        # open each tab and capture the treatment one
        page.get_by_role("tab", name="Treatment").click()
        page.wait_for_timeout(400)
        page.screenshot(path=str(OUT / "step2b_treatment_tab.png"), full_page=True)
        print("  treatment tab:", "Remove infected" in page.inner_text("body") or "fungicide" in page.inner_text("body"))

        # ============ STEP 3: low confidence ============
        page.get_by_text("Choose a different photo").click()
        page.wait_for_timeout(300)
        page.set_input_files("input[type=file]:not([capture])", str(LOW_CONF))
        page.wait_for_selector("text=Not confident about this one", timeout=60000)
        page.wait_for_timeout(1000)
        page.screenshot(path=str(OUT / "step3_low_confidence.png"), full_page=True)
        body = page.inner_text("body")
        print("\nSTEP 3 low confidence (leaf_spot_410)")
        print("  'not confident' state :", "Not confident about this one" in body)
        print("  'try clearer photo'   :", "single leaf or a single fruit" in body)
        print("  no result card leaked:", "TOP MATCHES" not in body.upper())
        # the UI renders this heading in uppercase (CSS text-transform), so match case-insensitively
        print("  shows alternatives   :", "what it leaned towards" in body.lower())
        ctx.close()

        # ============ STEP 4: powdery mildew merged ============
        ctx = b.new_context(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(700)
        page.set_input_files("input[type=file]:not([capture])", str(PM_LEAF))
        wait_result(page)
        page.screenshot(path=str(OUT / "step4_powdery_mildew_merged.png"), full_page=True)
        body = page.inner_text("body")
        print("\nSTEP 4 powdery mildew merged (powdery_mildew_leaf)")
        print("  'Powdery Mildew - detected on leaf':", "Powdery Mildew - detected on leaf" in body)
        print("  one-disease note shown  :", "one disease" in body and "single fungus" in body)
        print("  not two separate diseases:", "Powdery Mildew - detected on fruit" not in body)
        ctx.close()

        # ============ STEP 4b: ambiguous PM-fruit shown as its own class (honest) ============
        ctx = b.new_context(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(700)
        page.set_input_files("input[type=file]:not([capture])", str(AMBIG))
        wait_result(page)
        page.screenshot(path=str(OUT / "step4b_ambiguous.png"), full_page=True)
        body = page.inner_text("body")
        pred_line = next((l for l in body.splitlines() if l.strip().startswith("Model class:")), "n/a")
        print("\nSTEP 4b ambiguous powdery-fruit image")
        print("  model class :", pred_line)
        print("  (documented label ambiguity - predicted gray_mold, true label powdery_mildew_fruit)")
        ctx.close()

        # ============ STEP 5: bad input (wrong type) ============
        ctx = b.new_context(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(700)
        bad = PROJECT / "demo_images" / "_tmp_notanimage.jpg"
        bad.write_bytes(b"this is plain text pretending to be a jpg" * 5)
        page.set_input_files("input[type=file]:not([capture])", str(bad))
        page.wait_for_timeout(1200)
        page.screenshot(path=str(OUT / "step5_bad_input.png"), full_page=True)
        body = page.inner_text("body")
        print("\nSTEP 5 bad input (.txt bytes renamed .jpg)")
        # backend returns HTTP 400 {"detail":"File is not a readable image."}; the UI shows
        # "Could not analyse that photo" + that message + a retry button.
        low = body.lower()
        print("  clean error heading :", "could not analyse that photo" in low)
        print("  backend message shown:", "not a readable image" in low)
        print("  retry offered       :", "try another photo" in low)
        print("  no stack trace      :", "Traceback" not in body and "  at " not in body)
        bad.unlink()
        ctx.close()

        # ============ STEP 6: mobile ============
        ctx = b.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=3, is_mobile=True, has_touch=True)
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(800)
        page.set_input_files("input[type=file]:not([capture])", str(CONFIDENT))
        wait_result(page)
        page.screenshot(path=str(OUT / "step6_mobile_result.png"), full_page=True)
        # check no horizontal overflow
        overflow = page.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
        print("\nSTEP 6 mobile 390px result")
        print("  horizontal overflow px:", overflow, "(0 = no sideways scroll)")
        print("  confidence bar visible:", "Above the 60% reliability line" in page.inner_text("body"))
        ctx.close()

        b.close()
    print(f"\nscreenshots -> {OUT}")


if __name__ == "__main__":
    main()
