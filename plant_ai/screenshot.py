"""
Drive the real frontend in Chromium and capture screenshots for Phase 5 evidence.

Captures:
  1. landing / empty state (mobile + desktop)
  2. loading state
  3. a confident result (gray_mold, ~0.97)
  4. the powdery-mildew group UI (detected on leaf)
  5. a low-confidence result (the real leaf_spot_410 case)
  6. error state (backend down)
  7. mobile confident result
"""

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT = Path(r"D:\ivp\plant_ai")
DATASET = PROJECT / "dataset"
SHOTS = PROJECT / "screenshots"
SHOTS.mkdir(exist_ok=True)
URL = "http://127.0.0.1:5173/"

import csv

# resolve real test images from the manifest instead of guessing filenames
TEST = [r for r in csv.DictReader((DATASET / "manifest.csv").open(encoding="utf-8"))
        if r["split"] == "test" and r["is_dup_copy"] == "False"]
BY_CLASS = {}
for r in TEST:
    BY_CLASS.setdefault(r["final_class"], []).append(DATASET / r["path"])


def first(cls):
    return BY_CLASS[cls][0]


# confident: gray_mold scored 0.9683 in the API test
CONFIDENT = first("gray_mold")
PM_LEAF = first("powdery_mildew_leaf")
# low-confidence: the real leaf_spot_410 case (top-1 was only 0.22)
LOW_CONF = DATASET / "test" / "leaf_spot" / "leaf_spot_410.jpg"
if not LOW_CONF.exists():
    LOW_CONF = first("leaf_spot")
print(f"confident  -> {CONFIDENT.name}")
print(f"pm leaf    -> {PM_LEAF.name}")
print(f"low conf   -> {LOW_CONF.name}")


def shoot(page, name, full=True):
    p = SHOTS / f"{name}.png"
    page.screenshot(path=str(p), full_page=full)
    print(f"  saved {p.name}")


def upload(page, path):
    page.set_input_files("input[type=file]:not([capture])", str(path))
    page.wait_for_timeout(400)


def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch()

        # ---------- desktop ----------
        ctx = browser.new_context(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(700)

        print("[desktop] landing page")
        shoot(page, "01_landing_desktop")

        print("[desktop] confident result (gray_mold)")
        upload(page, CONFIDENT)
        page.wait_for_timeout(220)
        shoot(page, "02_loading", full=False)
        try:
            page.wait_for_selector("text=Top matches", timeout=45000)
        except Exception:
            print("   ! result did not render")
        page.wait_for_timeout(1400)
        shoot(page, "03_confident_result")

        for tab in ("Cause & spread", "Treatment", "Prevention"):
            try:
                page.get_by_role("tab", name=tab).click()
                page.wait_for_timeout(350)
            except Exception:
                print(f"   ! tab {tab} not found")
        shoot(page, "04_treatment_tab")

        print("[desktop] powdery mildew group UI")
        upload(page, PM_LEAF)
        try:
            page.wait_for_selector("text=Top matches", timeout=45000)
        except Exception:
            print("   ! result did not render")
        page.wait_for_timeout(1500)
        shoot(page, "05_powdery_mildew_group")

        print("[desktop] known limitations panel open")
        try:
            page.get_by_text("Known limitations of this model").click()
            page.wait_for_timeout(350)
            shoot(page, "06_limitations_open", full=False)
        except Exception as exc:
            print("   ! limitations:", exc)

        print("[desktop] low-confidence result")
        upload(page, LOW_CONF)
        try:
            page.wait_for_selector("text=Not confident about this one", timeout=45000)
        except Exception:
            print("   ! low-confidence panel did not render")
        page.wait_for_timeout(1500)
        shoot(page, "07_low_confidence")
        ctx.close()

        # ---------- mobile ----------
        mctx = browser.new_context(
            viewport={"width": 390, "height": 844}, device_scale_factor=3, is_mobile=True,
            has_touch=True,
        )
        mpage = mctx.new_page()
        mpage.goto(URL, wait_until="networkidle")
        mpage.wait_for_timeout(600)
        print("[mobile] landing")
        shoot(mpage, "08_landing_mobile")
        print("[mobile] confident result")
        upload(mpage, CONFIDENT)
        try:
            mpage.wait_for_selector("text=Top matches", timeout=45000)
        except Exception:
            print("   ! result did not render")
        mpage.wait_for_timeout(1400)
        shoot(mpage, "09_confident_mobile")
        mctx.close()

        # ---------- error state: backend unreachable ----------
        print("[error] backend down")
        ectx = browser.new_context(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
        epage = ectx.new_page()
        epage.route("**/api/**", lambda route: route.abort())
        epage.goto(URL, wait_until="networkidle")
        epage.wait_for_timeout(500)
        upload(epage, CONFIDENT)
        epage.wait_for_timeout(2500)
        shoot(epage, "10_error_backend_down")
        ectx.close()

        browser.close()
    print("\nall screenshots in", SHOTS)


if __name__ == "__main__":
    main()
