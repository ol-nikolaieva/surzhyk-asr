"""Нарізка сирих аудіо на кліпи 10 c, wav 16kHz mono (формат Whisper). Потрібен ffmpeg."""
import subprocess
from pathlib import Path

RAW, OUT = Path("data/raw"), Path("data/processed")
EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".mp4", ".webm"}

for f in RAW.iterdir():
    if f.suffix.lower() not in EXTS:
        continue
    subprocess.run([
        "ffmpeg", "-y", "-i", str(f),
        "-ac", "1", "-ar", "16000",          # mono, 16kHz
        "-f", "segment", "-segment_time", "10",  # різати по 10 c
        str(OUT / f"{f.stem}_%03d.wav"),
    ], check=True)

print("done:", len(list(OUT.glob("*.wav"))), "clips")
