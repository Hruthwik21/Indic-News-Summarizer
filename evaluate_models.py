"""
Compare summarization models (e.g. base XL-Sum mT5 vs your fine-tuned checkpoint) on ROUGE.

Examples:
  # XL-Sum test split, per language
  python evaluate_models.py --langs hindi,telugu,tamil \
      --models csebuetnlp/mT5_multilingual_XLSum,./checkpoints/indic-sum

  # Your own held-out file (CSV/JSON/JSONL with "text" and "summary" columns)
  python evaluate_models.py --data_file my_test.jsonl \
      --models csebuetnlp/mT5_multilingual_XLSum,./checkpoints/indic-sum

Outputs a table to the console plus eval_results.json (scores) and eval_samples.json
(a few example summaries per model, for eyeballing quality).

Run it in the training environment (datasets<4.0, recent transformers).
"""
import argparse
import json
import time

import evaluate
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="csebuetnlp/mT5_multilingual_XLSum,./checkpoints/indic-sum",
                   help="comma-separated model names or local paths")
    p.add_argument("--langs", default="hindi,telugu,tamil")
    p.add_argument("--data_file", default=None, help="evaluate on your own CSV/JSON/JSONL instead")
    p.add_argument("--n", type=int, default=200, help="examples per language / per file")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max_in", type=int, default=512)
    p.add_argument("--max_out", type=int, default=84)
    p.add_argument("--beams", type=int, default=4)
    p.add_argument("--samples", type=int, default=3, help="example outputs to save per group")
    p.add_argument("--out", default="eval_results.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def clean(s: str) -> str:
    return " ".join(s.split())


def load_groups(a):
    """Return {group_name: (texts, references)}."""
    groups = {}
    if a.data_file:
        kind = "json" if a.data_file.endswith((".json", ".jsonl")) else "csv"
        ds = load_dataset(kind, data_files=a.data_file, split="train")
        ds = ds.shuffle(seed=a.seed).select(range(min(a.n, len(ds))))
        groups["custom"] = ([clean(t) for t in ds["text"]], [clean(s) for s in ds["summary"]])
    else:
        for lang in [x.strip() for x in a.langs.split(",") if x.strip()]:
            # XL-Sum is script-based: needs trust_remote_code and datasets<4.0
            ds = load_dataset("csebuetnlp/xlsum", lang, split="test", trust_remote_code=True)
            ds = ds.shuffle(seed=a.seed).select(range(min(a.n, len(ds))))
            groups[lang] = ([clean(t) for t in ds["text"]], [clean(s) for s in ds["summary"]])
    return groups


@torch.inference_mode()
def generate_all(tok, model, texts, a):
    outputs = []
    for i in range(0, len(texts), a.batch):
        batch = texts[i:i + a.batch]
        enc = tok(batch, max_length=a.max_in, truncation=True, padding=True,
                  return_tensors="pt").to(DEVICE)
        ids = model.generate(**enc, max_length=a.max_out, num_beams=a.beams, no_repeat_ngram_size=2)
        outputs.extend(clean(s) for s in tok.batch_decode(ids, skip_special_tokens=True))
    return outputs


def main():
    a = parse_args()
    groups = load_groups(a)
    rouge = evaluate.load("rouge")
    results, samples = {}, {}

    for name in [m.strip() for m in a.models.split(",") if m.strip()]:
        print(f"\n=== {name} ===")
        tok = AutoTokenizer.from_pretrained(name)
        # fp32 on purpose: mT5 produces NaNs in fp16
        model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=torch.float32).to(DEVICE).eval()
        results[name], samples[name] = {}, {}

        for group, (texts, refs) in groups.items():
            t0 = time.time()
            preds = generate_all(tok, model, texts, a)
            secs = time.time() - t0
            # whitespace tokenization: the default ROUGE tokenizer discards non-Latin scripts
            r = rouge.compute(predictions=preds, references=refs,
                              tokenizer=lambda x: x.split(), use_stemmer=False)
            scores = {k: round(v * 100, 2) for k, v in r.items() if k in ("rouge1", "rouge2", "rougeL")}
            scores["avg_words"] = round(sum(len(p.split()) for p in preds) / max(len(preds), 1), 1)
            scores["sec_per_example"] = round(secs / max(len(preds), 1), 2)
            results[name][group] = scores
            samples[name][group] = [
                {"summary": preds[i], "reference": refs[i]} for i in range(min(a.samples, len(preds)))
            ]
            print(f"  {group:<10} {scores}")

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ---- comparison table ----
    print("\n" + "=" * 78)
    print(f"{'model':<42}{'group':<10}{'R1':>7}{'R2':>7}{'RL':>7}{'words':>7}")
    print("-" * 78)
    for model_name, per_group in results.items():
        short = model_name if len(model_name) <= 40 else "..." + model_name[-37:]
        for group, s in per_group.items():
            print(f"{short:<42}{group:<10}{s['rouge1']:>7}{s['rouge2']:>7}{s['rougeL']:>7}{s['avg_words']:>7}")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "scores": results}, f, ensure_ascii=False, indent=2)
    with open("eval_samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {a.out} and eval_samples.json")


if __name__ == "__main__":
    main()
