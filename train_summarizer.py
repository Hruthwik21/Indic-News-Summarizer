"""
Fine-tune a multilingual summarizer (mT5) for Indic news.

Data options:
  1. XL-Sum (BBC news, human-written summaries) for hindi, telugu, tamil, bengali,
     marathi, gujarati, punjabi, ...           -> --langs hindi,telugu,tamil
  2. Your own data (CSV or JSONL with "text" and "summary" columns)
                                               -> --data_file my_news.jsonl

Examples:
  # Small, fast run on Colab T4 with LoRA
  python train_summarizer.py --base google/mt5-small --langs hindi,telugu,tamil --lora

  # Adapt the already-strong XL-Sum model to your own news domain (recommended)
  python train_summarizer.py --base csebuetnlp/mT5_multilingual_XLSum \
      --data_file my_news.jsonl --epochs 2 --lr 1e-4

Then run the app with:  DIRECT_SUMMARIZER=./checkpoints/indic-sum python app.py

Install training deps first:  pip install -r requirements-train.txt
(Use a separate environment from the app: the app pins transformers==4.38.2.)
"""
import argparse
import inspect

import evaluate
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="google/mt5-small")
    p.add_argument("--langs", default="hindi,telugu,tamil,bengali,marathi")
    p.add_argument("--data_file", default=None, help="CSV/JSON/JSONL with text,summary columns")
    p.add_argument("--train_per_lang", type=int, default=3000)
    p.add_argument("--eval_per_lang", type=int, default=300)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max_in", type=int, default=512)
    p.add_argument("--max_out", type=int, default=84)
    p.add_argument("--lora", action="store_true", help="LoRA instead of full fine-tuning")
    p.add_argument("--out", default="./checkpoints/indic-sum")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_xlsum(langs, split, n, seed):
    parts = []
    for lang in langs:
        # XL-Sum is a script-based dataset: needs trust_remote_code and datasets<4.0
        ds = load_dataset("csebuetnlp/xlsum", lang, split=split, trust_remote_code=True)
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        parts.append(ds.select_columns(["text", "summary"]))
    return concatenate_datasets(parts).shuffle(seed=seed)


def load_custom(path, seed):
    kind = "json" if path.endswith((".json", ".jsonl")) else "csv"
    ds = load_dataset(kind, data_files=path, split="train").select_columns(["text", "summary"])
    split = ds.train_test_split(test_size=0.1, seed=seed)
    held = split["test"].train_test_split(test_size=0.5, seed=seed)
    return split["train"], held["train"], held["test"]  # train / validation / test


def main():
    a = parse_args()
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.base)

    if a.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            task_type="SEQ_2_SEQ_LM", r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q", "v"],
        ))
        model.print_trainable_parameters()

    # ---- data ----
    if a.data_file:
        train_raw, val_raw, test_raw = load_custom(a.data_file, a.seed)
    else:
        langs = [x.strip() for x in a.langs.split(",") if x.strip()]
        train_raw = load_xlsum(langs, "train", a.train_per_lang, a.seed)
        val_raw = load_xlsum(langs, "validation", a.eval_per_lang, a.seed)
        test_raw = load_xlsum(langs, "test", a.eval_per_lang, a.seed)

    def preprocess(batch):
        inputs = [" ".join(t.split()) for t in batch["text"]]
        targets = [" ".join(s.split()) for s in batch["summary"]]
        enc = tok(inputs, max_length=a.max_in, truncation=True)
        enc["labels"] = tok(text_target=targets, max_length=a.max_out, truncation=True)["input_ids"]
        return enc

    def prep(ds):
        return ds.map(preprocess, batched=True, remove_columns=ds.column_names)

    train_ds, val_ds, test_ds = prep(train_raw), prep(val_raw), prep(test_raw)

    # ---- metrics: whitespace-tokenized ROUGE (the default tokenizer drops non-Latin text) ----
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        dec_p = tok.batch_decode(preds, skip_special_tokens=True)
        dec_l = tok.batch_decode(labels, skip_special_tokens=True)
        r = rouge.compute(predictions=dec_p, references=dec_l,
                          tokenizer=lambda x: x.split(), use_stemmer=False)
        return {k: round(v * 100, 2) for k, v in r.items()}

    # ---- training args (handle the eval_strategy rename across transformers versions) ----
    strat_key = ("eval_strategy"
                 if "eval_strategy" in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
                 else "evaluation_strategy")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()  # mT5 + fp16 = NaNs

    kwargs = dict(
        output_dir=a.out,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch * 2,
        gradient_accumulation_steps=a.accum,
        learning_rate=a.lr,
        num_train_epochs=a.epochs,
        weight_decay=0.01,
        optim="adafactor",
        predict_with_generate=True,
        generation_max_length=a.max_out,
        generation_num_beams=4,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        bf16=use_bf16,
        report_to="none",
        seed=a.seed,
    )
    kwargs[strat_key] = "epoch"
    if not a.lora:
        kwargs.update(load_best_model_at_end=True, metric_for_best_model="rougeL")

    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(**kwargs),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Test metrics:", trainer.evaluate(test_ds, metric_key_prefix="test"))

    final = trainer.model
    if a.lora:
        final = final.merge_and_unload()  # so the app can load it like a normal model
    final.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"Saved to {a.out}. Run the app with DIRECT_SUMMARIZER={a.out}")


if __name__ == "__main__":
    main()
