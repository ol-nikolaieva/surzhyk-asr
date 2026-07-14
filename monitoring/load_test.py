"""Послідовне навантаження /transcribe для демонстрації метрик у Grafana."""
import time
from pathlib import Path

import requests

URL = "http://localhost:8000/transcribe"
DATA_DIR = Path("data/processed")
LIMIT = 20
PAUSE_SEC = 2

files = sorted(DATA_DIR.glob("*.wav"))[:LIMIT]
if not files:
    raise SystemExit(f"немає wav-файлів у {DATA_DIR}")

durations = []

for i, f in enumerate(files, start=1):
    start = time.perf_counter()
    with open(f, "rb") as fh:
        resp = requests.post(URL, files={"file": (f.name, fh, "audio/wav")})
    elapsed = time.perf_counter() - start
    durations.append(elapsed)

    print(f"[{i}/{len(files)}] {f.name} -> {resp.status_code} ({elapsed:.2f}s)")

    if i < len(files):
        time.sleep(PAUSE_SEC)

print()
print("--- підсумок ---")
print("всього запитів:", len(durations))
print(f"середній час: {sum(durations) / len(durations):.2f}s")
print(f"мін/макс: {min(durations):.2f}s / {max(durations):.2f}s")
