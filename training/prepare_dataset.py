"""Label Studio JSON -> CSV пар (шлях до wav, транскрипція)."""
import json, csv
from pathlib import Path
from urllib.parse import unquote

ANNOTATIONS = Path("data/labeled/annotations.json")
PROCESSED = Path("data/processed")
OUT = Path("training/dataset.csv")

# індекс реальних файлів: нормалізоване ім'я (пробіли -> _) -> реальний шлях
real_files = {p.name.replace(" ", "_"): p for p in PROCESSED.glob("*.wav")}

tasks = json.loads(ANNOTATIONS.read_text())
rows, missing = [], []

for t in tasks:
    if not t.get("annotations"):
        continue
    ann = t["annotations"][0]
    if ann.get("was_cancelled"):          # Skip-нуті кліпи
        continue
    texts = [r["value"]["text"][0]
             for r in ann["result"] if r["type"] == "textarea"]
    if not texts:
        continue

    # LS шлях: /data/upload/1/<hash>-<sanitized_name>.wav
    ls_name = unquote(Path(t["data"]["audio"]).name)   # %20 -> пробіл, якщо є
    ls_norm = ls_name.replace(" ", "_")                # єдиний вигляд для порівняння
    match = next((p for name, p in real_files.items()
                  if ls_norm.endswith(name)), None)
    if match is None:
        missing.append(ls_name)
        continue
    rows.append({"audio": str(match), "text": texts[0].strip()})

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["audio", "text"])
    w.writeheader()
    w.writerows(rows)

print(f"ok: {len(rows)} pairs, missing: {len(missing)}")
if missing:
    print("приклади незнайдених:", missing[:3])