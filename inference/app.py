"""FastAPI сервіс транскрипції суржику моделлю surzhyk-whisper@champion (MLflow Model Registry)."""
import os
import subprocess
import tempfile
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import mlflow.transformers
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from mlflow.exceptions import MlflowException
from transformers import Pipeline, pipeline

os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")

MODEL_URI = "models:/surzhyk-whisper@champion"
MODEL_LABEL = "surzhyk-whisper@champion"

state: dict = {}


def _load_champion(device: str) -> Pipeline:
    try:
        loaded = mlflow.transformers.load_model(MODEL_URI, device=device)
    except MlflowException:
        loaded = mlflow.transformers.load_model(MODEL_URI, return_type="components", device=device)

    if isinstance(loaded, Pipeline):
        return loaded

    return pipeline(
        task="automatic-speech-recognition",
        model=loaded["model"],
        tokenizer=loaded["tokenizer"],
        feature_extractor=loaded.get("feature_extractor"),
        device=device,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {MODEL_LABEL} from {tracking_uri} onto {device}...")
    state["asr"] = _load_champion(device)
    print("Model loaded.")

    yield

    state.clear()


app = FastAPI(lifespan=lifespan)


def _to_wav_16k_mono(src_path: str, dst_path: str) -> None:
    # та сама нормалізація, що й scripts/preprocess.py: mono, 16kHz
    subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000", dst_path],
        check=True, capture_output=True,
    )


def _read_wav(path: str) -> tuple[np.ndarray, float]:
    with wave.open(path, "rb") as wf:
        n_frames = wf.getnframes()
        sample_rate = wf.getframerate()
        raw = wf.readframes(n_frames)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, n_frames / sample_rate


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    asr = state.get("asr")
    if asr is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    with tempfile.TemporaryDirectory() as tmp_dir:
        src_suffix = Path(file.filename or "").suffix or ".audio"
        src_path = os.path.join(tmp_dir, f"input{src_suffix}")
        wav_path = os.path.join(tmp_dir, "normalized.wav")

        with open(src_path, "wb") as f:
            f.write(await file.read())

        try:
            _to_wav_16k_mono(src_path, wav_path)
        except subprocess.CalledProcessError as e:
            raise HTTPException(
                status_code=400,
                detail=f"ffmpeg не зміг обробити аудіо: {e.stderr.decode(errors='replace')}",
            ) from e

        audio, duration_sec = _read_wav(wav_path)
        result = asr({"raw": audio, "sampling_rate": 16000})

    text = result["text"].strip() if isinstance(result, dict) else str(result).strip()
    return {"text": text, "duration_sec": duration_sec}


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_LABEL}
