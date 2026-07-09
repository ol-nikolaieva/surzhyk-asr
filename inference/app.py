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
from fastapi.responses import HTMLResponse
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


INDEX_HTML = """<!doctype html>
<html lang="uk">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Surzhyk ASR</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #121212;
    --surface: #1c1c1e;
    --border: #2e2e30;
    --text: #ececec;
    --text-dim: #9a9a9a;
    --accent: #6ea8fe;
    --error: #f28b82;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    min-height: 100vh;
    display: flex;
    justify-content: center;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  main {
    width: 100%;
    max-width: 600px;
    padding: 48px 24px;
  }
  h1 {
    font-size: 1.6rem;
    margin: 0 0 4px;
  }
  .subtitle {
    color: var(--text-dim);
    margin: 0 0 32px;
  }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
  }
  input[type="file"] {
    width: 100%;
    color: var(--text-dim);
  }
  .file-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
  }
  .file-row input[type="file"] {
    flex: 1;
    min-width: 0;
  }
  .clear-btn {
    width: auto;
    flex-shrink: 0;
    padding: 8px 12px;
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-dim);
    font-weight: 400;
    border-radius: 8px;
  }
  .clear-btn:hover {
    color: var(--text);
    border-color: var(--text-dim);
  }
  #preview {
    width: 100%;
    margin-bottom: 16px;
  }
  button {
    width: 100%;
    padding: 12px;
    border: none;
    border-radius: 8px;
    background: var(--accent);
    color: #10151c;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
  #status {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 20px;
    color: var(--text-dim);
  }
  .spinner {
    width: 18px;
    height: 18px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  #result {
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
  }
  #result .text {
    white-space: pre-wrap;
    line-height: 1.5;
  }
  #result .meta {
    margin-top: 8px;
    color: var(--text-dim);
    font-size: 0.9rem;
  }
  #error {
    margin-top: 20px;
    color: var(--error);
  }
  [hidden] { display: none !important; }
</style>
</head>
<body>
<main>
  <h1>Surzhyk ASR</h1>
  <p class="subtitle">Транскрипція суржику</p>

  <div class="card">
    <form id="form">
      <div class="file-row">
        <input id="file" type="file" accept="audio/*" required>
        <button id="clear" type="button" class="clear-btn" hidden aria-label="Прибрати файл">✕</button>
      </div>
      <audio id="preview" controls hidden></audio>
      <button id="submit" type="submit">Транскрибувати</button>
    </form>

    <div id="status" hidden>
      <div class="spinner"></div>
      <span>Триває обробка...</span>
    </div>

    <div id="result" hidden>
      <div class="text" id="text"></div>
      <div class="meta">Тривалість: <span id="duration"></span></div>
    </div>

    <div id="error" hidden>От халепа, спробуй ще раз</div>
  </div>
</main>

<script>
  const form = document.getElementById('form');
  const fileInput = document.getElementById('file');
  const clearBtn = document.getElementById('clear');
  const preview = document.getElementById('preview');
  const statusEl = document.getElementById('status');
  const resultEl = document.getElementById('result');
  const errorEl = document.getElementById('error');
  const submitBtn = document.getElementById('submit');

  let previewUrl = null;

  fileInput.addEventListener('change', () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }

    if (fileInput.files.length) {
      previewUrl = URL.createObjectURL(fileInput.files[0]);
      preview.src = previewUrl;
      preview.hidden = false;
      clearBtn.hidden = false;
    } else {
      preview.src = '';
      preview.hidden = true;
      clearBtn.hidden = true;
    }
  });

  clearBtn.addEventListener('click', () => {
    fileInput.value = '';

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
    preview.src = '';
    preview.hidden = true;
    clearBtn.hidden = true;

    resultEl.hidden = true;
    errorEl.hidden = true;
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!fileInput.files.length) return;

    errorEl.hidden = true;
    resultEl.hidden = true;
    statusEl.hidden = false;
    submitBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
      const resp = await fetch('/transcribe', { method: 'POST', body: formData });
      if (!resp.ok) throw new Error('request failed');
      const data = await resp.json();

      document.getElementById('text').textContent = data.text;
      document.getElementById('duration').textContent = data.duration_sec.toFixed(2) + ' с';
      resultEl.hidden = false;
    } catch (err) {
      errorEl.hidden = false;
    } finally {
      statusEl.hidden = true;
      submitBtn.disabled = false;
    }
  });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


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
