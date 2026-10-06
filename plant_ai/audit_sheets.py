"""Step 3: build labelled contact sheets of flagged images for visual inspection.

One sheet per class, 8 columns, each tile numbered. The number is what I cite in the report.
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw

DS = Path(r"D:\ivp\plant_ai\dataset\train")
OUT = Path(r"D:\ivp\plant_ai\audit_sheets")
OUT.mkdir(exist_ok=True)

data = json.loads(Path(r"D:\ivp\plant_ai\audit_labels_stage2.json").read_text(encoding="utf-8"))

COLS, TILE = 8, 190
index = {}
for cls, v in sorted(data.items()):
    flagged = v["flagged"]
    if not flagged:
        continue
    rows = (len(flagged) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * TILE, rows * (TILE + 22)), "white")
    d = ImageDraw.Draw(sheet)
    order = sorted(flagged, key=lambda x: -x[1]["brown"])
    index[cls] = []
    for i, (name, s) in enumerate(order):
        p = DS / cls / name
        im = Image.open(p).convert("RGB").resize((TILE, TILE))
        x, y = (i % COLS) * TILE, (i // COLS) * (TILE + 22)
        sheet.paste(im, (x, y))
        d.rectangle([x, y + TILE, x + TILE, y + TILE + 22], fill="black")
        d.text((x + 4, y + TILE + 6), f"[{i+1}]", fill="yellow")
        d.text((x + 34, y + TILE + 6), name.replace(".jpg", "")[:20], fill="white")
        index[cls].append({"n": i + 1, "file": name,
                           "cov": round(s["coverage"], 3), "lum": round(s["lum"], 3),
                           "rough": round(s["rough"], 2), "brown": round(s["brown"], 2)})
    n_sheet = 1
    per = 64
    for start in range(0, len(order), per):
        chunk = order[start:start + per]
        r = (len(chunk) + COLS - 1) // COLS
        sh = Image.new("RGB", (COLS * TILE, r * (TILE + 22)), "white")
        dd = ImageDraw.Draw(sh)
        for j, (name, s) in enumerate(chunk):
            im = Image.open(DS / cls / name).convert("RGB").resize((TILE, TILE))
            x, y = (j % COLS) * TILE, (j // COLS) * (TILE + 22)
            sh.paste(im, (x, y))
            dd.rectangle([x, y + TILE, x + TILE, y + TILE + 22], fill="black")
            dd.text((x + 4, y + TILE + 6), f"[{start + j + 1}]", fill="yellow")
            dd.text((x + 34, y + TILE + 6), name.replace(".jpg", "")[:20], fill="white")
        sh.save(OUT / f"{cls}_sheet{n_sheet}.png")
        n_sheet += 1
    print(f"{cls:24s} {len(flagged):4d} flagged -> {n_sheet} sheet(s)")

Path(OUT / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
print("\nindex written to", OUT / "index.json")