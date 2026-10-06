"""Full visual sweep of the two classes whose rival-scan under-covered them.

anthracnose_fruit_rot has 50% test accuracy (worst class, 4/8 wrong) but the rival scan only
flagged 5 of 58 train images - a mismatch worth checking directly. Same for the images that
gray_mold was confused with. These sheets contain EVERY train image, not just flagged ones.
"""
from pathlib import Path

from PIL import Image, ImageDraw

DS = Path(r"D:\ivp\plant_ai\dataset\train")
OUT = Path(r"D:\ivp\plant_ai\audit_sheets")

for cls, cols in [("anthracnose_fruit_rot", 8), ("gray_mold", 8)]:
    files = [p for p in sorted((DS / cls).glob("*.jpg")) if "_hard" not in p.stem]
    TILE, per = 200, 64
    n_sheet = 0
    for start in range(0, len(files), per):
        chunk = files[start:start + per]
        rows = (len(chunk) + cols - 1) // cols
        sh = Image.new("RGB", (cols * TILE, rows * (TILE + 22)), "white")
        d = ImageDraw.Draw(sh)
        for j, p in enumerate(chunk):
            im = Image.open(p).convert("RGB").resize((TILE, TILE))
            x, y = (j % cols) * TILE, (j // cols) * (TILE + 22)
            sh.paste(im, (x, y))
            d.rectangle([x, y + TILE, x + TILE, y + TILE + 22], fill="black")
            d.text((x + 4, y + TILE + 6), f"[{start + j + 1}]", fill="yellow")
            d.text((x + 34, y + TILE + 6), p.stem[:20], fill="white")
        n_sheet += 1
        sh.save(OUT / f"ALL_{cls}_sheet{n_sheet}.png")
        print(f"{cls} sheet{n_sheet}: images {start+1}-{start+len(chunk)} of {len(files)}")
