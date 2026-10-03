"""Build the BLINDED human-audit pack for Task 3.

    python task3_gan/audit/make_audit_pack.py

Collects each member's Monet-style translations of the 30 team-fixed audit photos
(task3_gan/<member>/outputs/audit/), shuffles all items together, renames them Q01, Q02, ...
and writes:
    pack/audit_sheet.html        <- raters open this (shows input + output, never the member)
    pack/items/                  <- the images
    ratings_rater1.csv, ratings_rater2.csv   <- each rater fills in ONE of these, independently
    key/key.csv                  <- item -> member mapping. Raters must NOT open it until scoring.
"""
import csv
import json
import random
import shutil
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
split = json.loads((REPO / "task3_gan" / "common" / "team_eval_split.json").read_text())
photo_dir = REPO / "task3_gan" / "data" / "photo_jpg"

members = sorted(d.name for d in (REPO / "task3_gan").iterdir()
                 if (d / "outputs" / "audit").is_dir() and any((d / "outputs" / "audit").iterdir()))
if not members:
    raise SystemExit("No member has outputs/audit/ yet — finish the Task 3 notebook(s) first.")

items = [(m, n) for m in members for n in split["audit_photos"]
         if (REPO / "task3_gan" / m / "outputs" / "audit" / f"{Path(n).stem}.jpg").exists()]
random.Random(2026).shuffle(items)

pack, keyd = HERE / "pack", HERE / "key"
shutil.rmtree(pack, ignore_errors=True)
(pack / "items").mkdir(parents=True)
keyd.mkdir(exist_ok=True)
rows_html, key_rows = [], []
for i, (m, n) in enumerate(items, 1):
    qid = f"Q{i:02d}"
    Image.open(photo_dir / n).convert("RGB").resize((256, 256)).save(pack / "items" / f"{qid}_input.jpg", quality=92)
    shutil.copy(REPO / "task3_gan" / m / "outputs" / "audit" / f"{Path(n).stem}.jpg", pack / "items" / f"{qid}_output.jpg")
    key_rows.append({"item_id": qid, "member": m, "photo": n})
    rows_html.append(f"<tr><td><b>{qid}</b></td><td><img src='items/{qid}_input.jpg'></td>"
                     f"<td><img src='items/{qid}_output.jpg'></td></tr>")

with open(keyd / "key.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["item_id", "member", "photo"]); w.writeheader(); w.writerows(key_rows)
for r in (1, 2):
    with open(HERE / f"ratings_rater{r}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["item_id", "style", "content", "artifacts", "comment"])
        for k in key_rows:
            w.writerow([k["item_id"], "", "", "", ""])

rubric = """<h2>Rating scale (integers 1–5)</h2><ul>
<li><b>style</b> — 1 = nothing Monet-like · 3 = some painterly colour/texture · 5 = convincingly Monet (brush strokes, soft light, palette)</li>
<li><b>content</b> — 1 = scene unrecognisable · 3 = main layout kept, details lost · 5 = every object and the layout preserved</li>
<li><b>artifacts</b> — 1 = severe (checkerboard, blotches, colour bleeding, noise) · 3 = noticeable · 5 = none visible</li></ul>
<p>Rate each item on its own. Do not compare across items, and do not discuss with the other rater.
Fill in <code>ratings_rater1.csv</code> or <code>ratings_rater2.csv</code>.</p>"""
html = (f"<html><head><meta charset='utf-8'><title>Task 3 blinded audit</title><style>img{{width:256px}} td{{padding:6px}}"
        f"</style></head><body><h1>Blinded audit — {len(items)} items</h1>{rubric}<table><tr><th>item</th><th>input photo</th>"
        f"<th>Monet-style output</th></tr>{''.join(rows_html)}</table></body></html>")
(pack / "audit_sheet.html").write_text(html, encoding="utf-8")
print(f"{len(items)} items from members {members}. Give raters task3_gan/audit/pack/ and one ratings CSV each. "
      f"Keep task3_gan/audit/key/ closed until scoring.")
