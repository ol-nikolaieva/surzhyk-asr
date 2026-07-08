"""Fine-tuning Whisper на суржиковому датасеті + MLflow трекінг."""
import argparse, os, dataclasses
import mlflow
import torch
import evaluate
from datasets import load_dataset, Audio
from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)

os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")

p = argparse.ArgumentParser()
p.add_argument("--model", default="openai/whisper-tiny")
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--dataset-version", default="v2")
args = p.parse_args()

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("surzhyk-whisper")

# --- дані ---
ds = load_dataset("csv", data_files="training/dataset.csv", split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
ds = ds.train_test_split(test_size=0.15, seed=42)

processor = WhisperProcessor.from_pretrained(args.model, language="ukrainian", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(args.model)
model.generation_config.language = "ukrainian"

def prepare(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(
        audio["array"], sampling_rate=16000).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

ds = ds.map(prepare, remove_columns=ds["train"].column_names)

# --- колатор ---
@dataclasses.dataclass
class Collator:
    def __call__(self, features):
        input_feats = [{"input_features": f["input_features"]} for f in features]
        batch = processor.feature_extractor.pad(input_feats, return_tensors="pt")
        labels = [{"input_ids": f["labels"]} for f in features]
        labels = processor.tokenizer.pad(labels, return_tensors="pt")
        batch["labels"] = labels["input_ids"].masked_fill(
            labels.attention_mask.ne(1), -100)
        return batch

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": cer_metric.compute(predictions=pred_str, references=label_str)}

train_args = Seq2SeqTrainingArguments(
    output_dir="training/checkpoints",
    per_device_train_batch_size=args.batch,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    eval_strategy="epoch",
    predict_with_generate=True,
    logging_steps=5,
    report_to=[],
    use_cpu=not torch.backends.mps.is_available(),
)

trainer = Seq2SeqTrainer(
    model=model, args=train_args,
    train_dataset=ds["train"], eval_dataset=ds["test"],
    data_collator=Collator(), compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

with mlflow.start_run():
    mlflow.log_params({"model": args.model, "epochs": args.epochs,
                       "lr": args.lr, "batch": args.batch,
                       "dataset_version": args.dataset_version,
                       "train_size": len(ds["train"]), "eval_size": len(ds["test"])})
    trainer.train()
    final = trainer.evaluate()
    mlflow.log_metrics({"final_wer": final["eval_wer"],
                        "final_cer": final["eval_cer"]})
    mlflow.transformers.log_model(
        transformers_model={"model": trainer.model, "tokenizer": processor.tokenizer,
                            "feature_extractor": processor.feature_extractor},
        name="model",
        task="automatic-speech-recognition",
        registered_model_name="surzhyk-whisper",
    )
    print("WER:", final["eval_wer"], "CER:", final["eval_cer"])
