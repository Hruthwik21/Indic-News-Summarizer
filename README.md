# Indic News Summarizer v2

Summarize a news article from a URL (or pasted text), written in English or a major Indian language, and read the summary in the language you choose. Version 2 improves extraction, translation quality, long-article handling and model choice, and adds a fine-tuning and evaluation workflow.

---

## Table of Contents

1. [What's New in v2](#whats-new-in-v2)
2. [Architecture](#architecture)
3. [Pipeline Details](#pipeline-details)
4. [Models](#models)
5. [Supported Languages](#supported-languages)
6. [Installation](#installation)
7. [Running the App](#running-the-app)
8. [Fine-Tuning](#fine-tuning)
9. [Evaluation](#evaluation)
10. [Configuration Reference](#configuration-reference)
11. [Error Handling](#error-handling)
12. [Known Limitations](#known-limitations)
13. [Project Structure](#project-structure)
14. [Credits](#credits)

---

## What's New in v2

| v1 | v2 |
|---|---|
| Only the first 3,500 characters processed | Long articles are summarized in chunks, then merged (up to `MAX_CHARS`, default 12,000) |
| Whole text translated in one pass | Sentence-level, batched translation (IndicTrans2 is a sentence-level model) |
| Scraped `<p>` tags only | `trafilatura` extraction, `<p>` scraping as fallback, plus a paste-text box |
| `langdetect` for every language | Unicode-script detection, with `langdetect` only separating Hindi from Marathi |
| All three models loaded at startup | Models load lazily and are cached |
| Single pivot-through-English route | `auto`, `pivot` and `direct` modes |
| No length control | Short / medium / long summaries |
| Token written in the code, `share=True` always | Token from `HF_TOKEN`, sharing opt-in via `GRADIO_SHARE=1` |
| No training or evaluation | `train_summarizer.py` and `evaluate_models.py` |
| Errors shown only as text | Diagnostics panel: language, route, chunks, time |

---

## Architecture

```
 URL or pasted text
        |
        v
 [1] Extract text  (trafilatura -> <p> fallback)
        |
        v
 [2] Detect language  (Unicode script; langdetect for hi vs mr)
        |
        +--------------------------+-------------------------------+
        | PIVOT                    | DIRECT                        |
        | (English or unsupported) | (language in DIRECT_LANGS)    |
        v                          v                               |
 translate to English        summarize in source language          |
 (IndicTrans2 indic-en,      (mT5 XL-Sum or your fine-tuned        |
  sentence by sentence)       checkpoint, chunked map-reduce)      |
        |                          |                               |
        v                          |                               |
 summarize in English              |                               |
 (BART-large-CNN,                  |                               |
  chunked map-reduce)              |                               |
        |                          |                               |
        v                          v                               |
 translate summary to target language (IndicTrans2) <--------------+
 (skipped when the summary is already in the target language)
        |
        v
   Gradio UI: summary + diagnostics
```

---

## Pipeline Details

### 1. Text extraction

- `curl_cffi` fetches the page while impersonating a Chrome browser (`IMPERSONATE`, default `chrome110`), which gets past basic bot blocking. Timeout: 20 seconds.
- `trafilatura` extracts the main article body and drops navigation, comments and tables.
- If that yields fewer than 300 characters, the fallback removes `script`, `style`, `nav`, `footer`, `aside`, `form` and `header` tags and keeps paragraphs longer than 40 characters.
- If the user pastes text, scraping is skipped entirely. This is the workaround for paywalls and blocked sites.
- Text under 300 characters is rejected with a helpful message. Text over `MAX_CHARS` is cut and the diagnostics panel says so.

### 2. Language detection

- Counts alphabetic characters per Unicode script (Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam). If a script covers more than 30% of letters, that script decides the language.
- Devanagari is shared by Hindi and Marathi, so `langdetect` (seeded for determinism) picks between them, defaulting to Hindi.
- Anything else is treated as English.

### 3. Route selection

| Mode | Behavior |
|---|---|
| `auto` | Direct if the detected language is in `DIRECT_LANGS`, otherwise pivot. English always uses pivot (BART) |
| `pivot` | Always translate to English, summarize with BART, translate back |
| `direct` | Summarize in the source language. If the language isn't supported, falls back to pivot and says so |

### 4. Translation (IndicTrans2)

- The text is split into sentences on `. ! ?` and the Devanagari danda marks (`।`, `॥`).
- Sentences are translated in batches of 16 with 5-beam search and a 256-token cap per sentence.
- Indic-to-Indic goes through English (`src -> eng_Latn -> tgt`). Source equal to target is a no-op.
- `IndicProcessor` handles pre- and post-processing for every batch.

### 5. Summarization

**Pivot (BART, English):** sentences are grouped into chunks of up to 900 tokens. A single chunk is summarized directly. For several chunks, each is summarized to 30-120 tokens (map), the partial summaries are joined, and a final pass summarizes that (reduce). Generation uses 4 beams, `length_penalty=2.0` and `no_repeat_ngram_size=3`.

**Direct (mT5):** same map-reduce idea with 480-token chunks, 84-token partial summaries, 4 beams and `no_repeat_ngram_size=2`. The model runs in fp32 because mT5 produces NaN values in fp16.

### 6. Length presets

| Preset | English (max / min tokens) | Direct (max tokens) |
|---|---|---|
| short | 80 / 25 | 64 |
| medium | 150 / 40 | 84 |
| long | 250 / 80 | 128 |

---

## Models

| Role | Default model | Override |
|---|---|---|
| Indic to English | `ai4bharat/indictrans2-indic-en-dist-200M` | constant in `app.py` |
| English summarizer | `facebook/bart-large-cnn` | `ENGLISH_SUMMARIZER` |
| Direct multilingual summarizer | `csebuetnlp/mT5_multilingual_XLSum` | `DIRECT_SUMMARIZER` (also accepts a local path to your fine-tuned model) |
| English to Indic | `ai4bharat/indictrans2-en-indic-dist-200M` | constant in `app.py` |

---

## Supported Languages

**Summary languages (dropdown):** Hindi, Telugu, Tamil, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Odia, Assamese, English.

**Direct mode (source languages):** Hindi, Marathi, Bengali, Gujarati, Punjabi, Tamil, Telugu. These are covered by XL-Sum. Kannada, Malayalam, Odia and Assamese are not, so they use the pivot route unless you fine-tune on your own data and add them through `DIRECT_LANGS`.

**Auto-detected source languages:** English plus the nine scripts above. Assamese is detected as Bengali because they share a script.

---

## Installation

```bash
git clone <your-repo>
cd indic_news_summarizer_v2
pip install -r requirements.txt
export HF_TOKEN=your_token_here   # only if a model download requires login
```

`transformers` is pinned to 4.38.2 because the IndicTrans2 remote code depends on it. A GPU is strongly recommended.

For fine-tuning and evaluation, use a **separate environment**:

```bash
pip install -r requirements-train.txt
```

---

## Running the App

```bash
python app.py
# optional
GRADIO_SHARE=1 python app.py                          # public temporary link
DIRECT_SUMMARIZER=./checkpoints/indic-sum python app.py   # use your fine-tuned model
```

In the UI: paste a URL or article text, choose the summary language, mode and length, and press **Summarize**. The diagnostics panel shows the detected language, route taken, number of chunks and time.

---

## Fine-Tuning

`train_summarizer.py` fine-tunes an mT5 summarizer with `Seq2SeqTrainer`.

```bash
# Quick run on a small model with LoRA (fits a Colab T4)
python train_summarizer.py --base google/mt5-small --langs hindi,telugu,tamil --lora

# Adapt the XL-Sum model to your own domain (recommended starting point)
python train_summarizer.py --base csebuetnlp/mT5_multilingual_XLSum \
    --data_file my_news.jsonl --epochs 2 --lr 1e-4
```

Key options:

| Flag | Default | Meaning |
|---|---|---|
| `--base` | `google/mt5-small` | Starting checkpoint |
| `--langs` | `hindi,telugu,tamil,bengali,marathi` | XL-Sum languages to train on |
| `--data_file` | none | Your CSV/JSON/JSONL with `text` and `summary` columns (replaces XL-Sum) |
| `--train_per_lang` / `--eval_per_lang` | 3000 / 300 | Subsample sizes per language |
| `--epochs`, `--lr` | 3, 5e-4 | Training length and learning rate |
| `--batch`, `--accum` | 4, 8 | Per-device batch size and gradient accumulation |
| `--max_in`, `--max_out` | 512, 84 | Token limits for article and summary |
| `--lora` | off | LoRA adapters instead of full fine-tuning; merged before saving |
| `--out` | `./checkpoints/indic-sum` | Output folder |

Notes:

- Uses Adafactor, and bf16 only when the GPU supports it (mT5 breaks in fp16).
- Metric is whitespace-tokenized ROUGE, since the default ROUGE tokenizer drops non-Latin text.
- XL-Sum needs `datasets<4.0` and `trust_remote_code=True`.
- A small model trained on a few thousand examples will usually score below the published XL-Sum mT5 model. Fine-tuning pays off most when adapting that model to your own data.

---

## Evaluation

`evaluate_models.py` compares models on ROUGE-1/2/L, average summary length and speed.

```bash
python evaluate_models.py --langs hindi,telugu,tamil \
    --models csebuetnlp/mT5_multilingual_XLSum,./checkpoints/indic-sum

python evaluate_models.py --data_file my_test.jsonl \
    --models csebuetnlp/mT5_multilingual_XLSum,./checkpoints/indic-sum
```

It prints a comparison table and writes `eval_results.json` (scores) and `eval_samples.json` (example summaries next to references). Always read the samples: ROUGE rewards word overlap and can't catch factual errors.

---

## Configuration Reference

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | unset | Hugging Face token for model downloads |
| `ENGLISH_SUMMARIZER` | `facebook/bart-large-cnn` | English summarization model |
| `DIRECT_SUMMARIZER` | `csebuetnlp/mT5_multilingual_XLSum` | Direct summarizer or fine-tuned path |
| `DIRECT_LANGS` | `hin_Deva,mar_Deva,ben_Beng,guj_Gujr,pan_Guru,tam_Taml,tel_Telu` | Languages that use direct mode |
| `MAX_CHARS` | `12000` | Article length cap |
| `IMPERSONATE` | `chrome110` | Browser fingerprint for scraping |
| `GRADIO_SHARE` | `0` | Set to `1` for a public link |

---

## Error Handling

- Missing input: asks for a URL or pasted text.
- Empty, blocked or too-short pages: suggests pasting the text instead.
- Any pipeline exception is caught and shown as `Error: <message>` in the output box rather than crashing the app.
- If direct mode is requested for an unsupported language, it falls back to pivot and notes this in the diagnostics.

---

## Known Limitations

- **Accuracy:** summaries are machine-generated and can contain errors or omissions, and translation can add more. Verify important facts against the source.
- **Language coverage:** direct mode covers only the XL-Sum Indic languages unless you fine-tune on more. Other-script or non-Indic foreign languages (such as French) are treated as English.
- **Assamese** is detected as Bengali.
- **Scraping:** some sites need JavaScript rendering or block automated access. Check each site's terms of service and robots.txt, and use the paste option where scraping isn't appropriate.
- **Sentence splitting** is rule-based and can mis-split around abbreviations.
- **Hardware:** several models can be resident at once, and CPU inference is slow.
- **Untested at release:** the code passed a syntax check but should be tried on your GPU environment before relying on it.

---

## Project Structure

```
indic_news_summarizer_v2/
├── app.py                  # Gradio app and full inference pipeline
├── train_summarizer.py     # Fine-tuning (XL-Sum or custom data, optional LoRA)
├── evaluate_models.py      # ROUGE comparison of base vs fine-tuned models
├── requirements.txt        # App dependencies (transformers pinned)
├── requirements-train.txt  # Training and evaluation dependencies
└── README.md
```

---

## Credits

- [AI4Bharat IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) and [IndicTransToolkit](https://github.com/VarunGumma/IndicTransToolkit)
- [BART-large-CNN](https://huggingface.co/facebook/bart-large-cnn) (Facebook AI)
- [mT5 multilingual XL-Sum](https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum) and the XL-Sum dataset (CSE, BUET)
- Hugging Face Transformers, Datasets, PEFT, Gradio and trafilatura
