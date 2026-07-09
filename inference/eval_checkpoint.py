"""Оцінка чекпоінта Whisper на eval-спліті (WER/CER) + таблиця помилок."""
import argparse

import torch
from datasets import load_dataset, Audio
from jiwer import wer, cer
from transformers import (WhisperFeatureExtractor, WhisperForConditionalGeneration,
                          WhisperProcessor, WhisperTokenizerFast)

# checkpoints saved by training/train.py only include the feature extractor
# (Trainer's processing_class was set to processor.feature_extractor, not the
# full processor), so there's no tokenizer to load from the checkpoint itself.
TOKENIZER_FALLBACK = "openai/whisper-tiny"

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="training/checkpoints/checkpoint-84")
p.add_argument("--dataset", default="training/dataset.csv")
args = p.parse_args()

device = "mps"

try:
    processor = WhisperProcessor.from_pretrained(args.checkpoint, language="ukrainian", task="transcribe")
except (TypeError, OSError):
    print(f"No tokenizer found in {args.checkpoint}, using tokenizer from {TOKENIZER_FALLBACK} "
          "(multilingual Whisper vocab is identical across model sizes).")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.checkpoint)
    tokenizer = WhisperTokenizerFast.from_pretrained(TOKENIZER_FALLBACK, language="ukrainian", task="transcribe")
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

model = WhisperForConditionalGeneration.from_pretrained(args.checkpoint).to(device)
model.eval()

# той самий спліт, що й у training/train.py, щоб eval-кліпи збігалися
ds = load_dataset("csv", data_files=args.dataset, split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
ds = ds.train_test_split(test_size=0.15, seed=42)
eval_ds = ds["test"]

references = []
predictions = []
rows = []

n = len(eval_ds)
for i, example in enumerate(eval_ds):
    print(f"[{i + 1}/{n}] processing clip...")

    audio = example["audio"]
    reference = example["text"]

    inputs = processor(audio["array"], sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        generated_ids = model.generate(input_features)

    prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    references.append(reference)
    predictions.append(prediction)
    rows.append((i, reference, prediction))

    print(f"  ref:  {reference}")
    print(f"  pred: {prediction}")

overall_wer = wer(references, predictions)
overall_cer = cer(references, predictions)

print("\n=== Фінальні метрики ===")
print(f"WER: {overall_wer:.4f}")
print(f"CER: {overall_cer:.4f}")

print("\n=== Таблиця eval-кліпів (ground truth vs predicted) ===")
for idx, reference, prediction in rows:
    print(f"\nclip #{idx}")
    print(f"  gt:   {reference}")
    print(f"  pred: {prediction}")
