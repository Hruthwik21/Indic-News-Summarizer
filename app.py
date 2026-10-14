"""
Indic News Summarizer v2
------------------------
Improvements over v1:
  * Better article extraction (trafilatura, falls back to <p> scraping) + paste-text option
  * Script-based language detection (fast, reliable for Indic) with langdetect only for hi/mr
  * Sentence-level translation in batches (IndicTrans2 is a sentence-level model)
  * Map-reduce summarization so long articles are not silently truncated
  * Two modes: "pivot" (translate -> English BART -> translate) and
    "direct" (summarize in the source language with a multilingual model, which can be
    your own fine-tuned checkpoint from train_summarizer.py)
  * Lazy model loading with caching (only loads what a request needs)
  * Summary length control, diagnostics panel, and no hard-coded secrets

Run:  python app.py
Env:  HF_TOKEN, DIRECT_SUMMARIZER, ENGLISH_SUMMARIZER, GRADIO_SHARE=1, MAX_CHARS
"""
from __future__ import annotations

import os
import re
import time
from collections import Counter
from functools import lru_cache

import gradio as gr
import torch
from bs4 import BeautifulSoup
from curl_cffi import requests as c_requests
from langdetect import DetectorFactory, detect
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:  # newer toolkit layout
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:  # older layout used in v1
    from IndicTransToolkit import IndicProcessor

try:
    import trafilatura
except ImportError:
    trafilatura = None

DetectorFactory.seed = 0  # make langdetect deterministic

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN")  # never hard-code tokens

INDIC_EN_MODEL = "ai4bharat/indictrans2-indic-en-dist-200M"
EN_INDIC_MODEL = "ai4bharat/indictrans2-en-indic-dist-200M"
ENGLISH_SUMMARIZER = os.getenv("ENGLISH_SUMMARIZER", "facebook/bart-large-cnn")
# Point this at your fine-tuned checkpoint (e.g. ./checkpoints/indic-sum) after training
DIRECT_SUMMARIZER = os.getenv("DIRECT_SUMMARIZER", "csebuetnlp/mT5_multilingual_XLSum")

# Languages the direct multilingual summarizer is trusted on (XL-Sum covers these Indic ones).
# If you fine-tune on more languages, extend this list via the DIRECT_LANGS env var.
DIRECT_LANGS = set(
    os.getenv(
        "DIRECT_LANGS",
        "hin_Deva,mar_Deva,ben_Beng,guj_Gujr,pan_Guru,tam_Taml,tel_Telu",
    ).split(",")
)

MAX_CHARS = int(os.getenv("MAX_CHARS", "12000"))  # cap on article length processed
MIN_CHARS = 300
IMPERSONATE = os.getenv("IMPERSONATE", "chrome110")

LANGS = {
    "Hindi": "hin_Deva", "Telugu": "tel_Telu", "Tamil": "tam_Taml",
    "Kannada": "kan_Knda", "Malayalam": "mal_Mlym", "Bengali": "ben_Beng",
    "Marathi": "mar_Deva", "Gujarati": "guj_Gujr", "Punjabi": "pan_Guru",
    "Odia": "ory_Orya", "Assamese": "asm_Beng", "English": "eng_Latn",
}
CODE_TO_NAME = {v: k for k, v in LANGS.items()}

# (english max, english min, direct max) tokens
LENGTH_PRESETS = {
    "short": (80, 25, 64),
    "medium": (150, 40, 84),
    "long": (250, 80, 128),
}

IP = IndicProcessor(inference=True)


# --------------------------------------------------------------------------- #
# Model loading (lazy + cached)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def load_model(name: str, half: bool, remote: bool):
    """Load a tokenizer/model pair once. mT5 must stay fp32 (fp16 gives NaNs)."""
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=remote, token=HF_TOKEN)
    dtype = torch.float16 if (half and DEVICE == "cuda") else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        name, trust_remote_code=remote, torch_dtype=dtype, token=HF_TOKEN
    ).to(DEVICE).eval()
    return tok, model


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_article(url: str) -> str:
    """Fetch a page like a browser and return clean article text."""
    resp = c_requests.get(url, impersonate=IMPERSONATE, timeout=20)
    resp.raise_for_status()
    html = resp.text

    text = ""
    if trafilatura is not None:
        text = trafilatura.extract(
            html, include_comments=False, include_tables=False, favor_recall=True
        ) or ""

    if len(text) < MIN_CHARS:  # fallback: paragraph scraping with junk removed
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside", "form", "header"]):
            tag.decompose()
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(p for p in paras if len(p) > 40)

    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Language detection (script-based, deterministic)
# --------------------------------------------------------------------------- #
SCRIPTS = [
    ("Deva", 0x0900, 0x097F), ("Beng", 0x0980, 0x09FF), ("Guru", 0x0A00, 0x0A7F),
    ("Gujr", 0x0A80, 0x0AFF), ("Orya", 0x0B00, 0x0B7F), ("Taml", 0x0B80, 0x0BFF),
    ("Telu", 0x0C00, 0x0C7F), ("Knda", 0x0C80, 0x0CFF), ("Mlym", 0x0D00, 0x0D7F),
]
SCRIPT_TO_LANG = {
    "Beng": "ben_Beng", "Guru": "pan_Guru", "Gujr": "guj_Gujr", "Orya": "ory_Orya",
    "Taml": "tam_Taml", "Telu": "tel_Telu", "Knda": "kan_Knda", "Mlym": "mal_Mlym",
}


def detect_language(text: str) -> str:
    """Return an IndicTrans2 code. Script decides; langdetect only splits Hindi/Marathi."""
    sample = text[:2000]
    counts, letters = Counter(), 0
    for ch in sample:
        if not ch.isalpha():
            continue
        letters += 1
        cp = ord(ch)
        for name, lo, hi in SCRIPTS:
            if lo <= cp <= hi:
                counts[name] += 1
                break
    if letters and counts:
        script, n = counts.most_common(1)[0]
        if n / letters > 0.3:
            if script == "Deva":
                try:
                    return "mar_Deva" if detect(sample) == "mr" else "hin_Deva"
                except Exception:
                    return "hin_Deva"
            # Note: Assamese shares the Bengali script and is detected as Bengali.
            return SCRIPT_TO_LANG[script]
    return "eng_Latn"


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
_SENT_SPLIT = re.compile(r"(?<=[.!?\u0964\u0965])\s+")  # includes the danda marks


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s for s in _SENT_SPLIT.split(text) if len(s) > 1]


def chunk_by_tokens(sentences: list[str], tok, budget: int) -> list[str]:
    """Group whole sentences into chunks that fit a token budget."""
    chunks, cur, n = [], [], 0
    for s in sentences:
        k = len(tok.tokenize(s))
        if cur and n + k > budget:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += k
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# --------------------------------------------------------------------------- #
# Translation (sentence-level, batched, Indic<->Indic goes through English)
# --------------------------------------------------------------------------- #
def translate(sentences: list[str], src: str, tgt: str, batch_size: int = 16) -> list[str]:
    if not sentences or src == tgt:
        return sentences
    if src != "eng_Latn" and tgt != "eng_Latn":
        return translate(translate(sentences, src, "eng_Latn", batch_size), "eng_Latn", tgt, batch_size)

    name = INDIC_EN_MODEL if tgt == "eng_Latn" else EN_INDIC_MODEL
    tok, model = load_model(name, True, True)

    out: list[str] = []
    for i in range(0, len(sentences), batch_size):
        batch = IP.preprocess_batch(sentences[i:i + batch_size], src_lang=src, tgt_lang=tgt)
        enc = tok(batch, truncation=True, padding="longest", max_length=256,
                  return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            gen = model.generate(**enc, use_cache=True, min_length=0, max_length=256,
                                 num_beams=5, num_return_sequences=1)
        decoded = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        out.extend(IP.postprocess_batch(decoded, lang=tgt))
    return out


def translate_text(text: str, src: str, tgt: str) -> str:
    return " ".join(translate(split_sentences(text), src, tgt))


# --------------------------------------------------------------------------- #
# Summarizers
# --------------------------------------------------------------------------- #
def _generate(tok, model, text: str, max_in: int, **gen_kwargs) -> str:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_in).to(DEVICE)
    with torch.inference_mode():
        ids = model.generate(**enc, **gen_kwargs)
    return tok.decode(ids[0], skip_special_tokens=True).strip()


def summarize_english(text: str, max_len: int, min_len: int) -> tuple[str, int]:
    """BART with map-reduce for long inputs. Returns (summary, num_chunks)."""
    tok, model = load_model(ENGLISH_SUMMARIZER, True, False)
    chunks = chunk_by_tokens(split_sentences(text), tok, 900)
    if not chunks:
        return "", 0

    def run(chunk: str, mx: int, mn: int) -> str:
        return _generate(tok, model, chunk, 1024, max_length=mx, min_length=min(mn, mx - 5),
                         num_beams=4, length_penalty=2.0, no_repeat_ngram_size=3,
                         early_stopping=True)

    if len(chunks) == 1:
        return run(chunks[0], max_len, min_len), 1
    partials = [run(c, 120, 30) for c in chunks]          # map
    return run(" ".join(partials), max_len, min_len), len(chunks)  # reduce


def summarize_direct(text: str, max_len: int) -> tuple[str, int]:
    """Multilingual mT5 (base XL-Sum model or your fine-tuned checkpoint)."""
    tok, model = load_model(DIRECT_SUMMARIZER, False, False)  # fp32: mT5 breaks in fp16
    chunks = chunk_by_tokens(split_sentences(text), tok, 480)
    if not chunks:
        return "", 0

    def run(chunk: str, mx: int) -> str:
        return _generate(tok, model, chunk, 512, max_length=mx, num_beams=4,
                         no_repeat_ngram_size=2)

    if len(chunks) == 1:
        return run(chunks[0], max_len), 1
    partials = [run(c, 84) for c in chunks]
    return run(" ".join(partials), max_len), len(chunks)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(url: str, pasted: str, target_name: str, mode: str, length: str):
    t0 = time.time()
    try:
        # 1. Get text
        if pasted and pasted.strip():
            raw, source = pasted.strip(), "pasted text"
        elif url and url.strip():
            raw, source = extract_article(url.strip()), "scraped URL"
        else:
            return "Please provide a URL or paste article text.", ""
        if len(raw) < MIN_CHARS:
            return ("Could not get enough text (blocked page, paywall, or very short article). "
                    "Try pasting the article text instead."), ""
        truncated = len(raw) > MAX_CHARS
        raw = raw[:MAX_CHARS]

        # 2. Detect language and choose route
        src = detect_language(raw)
        tgt = LANGS[target_name]
        en_max, en_min, direct_max = LENGTH_PRESETS[length]

        use_direct = mode == "direct" or (mode == "auto" and src in DIRECT_LANGS)
        note = ""
        if src == "eng_Latn":
            use_direct = False
        elif use_direct and src not in DIRECT_LANGS:
            use_direct = False
            note = " (direct model doesn't cover this language, used pivot instead)"

        # 3. Summarize
        if use_direct:
            summary_src, n_chunks = summarize_direct(raw, direct_max)
            final = translate_text(summary_src, src, tgt)  # no-op if src == tgt
            route = "direct (source-language summary" + ("" if src == tgt else " -> translated") + ")"
        else:
            if src == "eng_Latn":
                english = raw
            else:
                english = " ".join(translate(split_sentences(raw), src, "eng_Latn"))
            summary_en, n_chunks = summarize_english(english, en_max, en_min)
            final = summary_en if tgt == "eng_Latn" else translate_text(summary_en, "eng_Latn", tgt)
            route = "pivot (translate -> English summary -> translate)"

        info = (
            f"**Source:** {source}  \n"
            f"**Detected language:** {CODE_TO_NAME.get(src, src)}  \n"
            f"**Route:** {route}{note}  \n"
            f"**Chunks summarized:** {n_chunks}"
            f"{'  (article was truncated to ' + str(MAX_CHARS) + ' chars)' if truncated else ''}  \n"
            f"**Time:** {time.time() - t0:.1f}s"
        )
        return final, info

    except Exception as exc:  # surface errors to the UI instead of crashing
        return f"Error: {exc}", ""


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
with gr.Blocks(title="Indic News Summarizer v2") as demo:
    gr.Markdown(
        "# Indic News Summarizer v2\n"
        "Summarize a news article from a URL (or pasted text) in English or an Indian "
        "language, and read the summary in the language you choose. Machine-generated "
        "summaries can contain errors, so verify important facts against the source."
    )
    with gr.Row():
        with gr.Column():
            url_in = gr.Textbox(label="News article URL", placeholder="https://...")
            text_in = gr.Textbox(label="...or paste article text (used if filled)", lines=6)
            target_in = gr.Dropdown(list(LANGS), value="Telugu", label="Summary language")
            mode_in = gr.Radio(["auto", "pivot", "direct"], value="auto", label="Mode")
            length_in = gr.Radio(["short", "medium", "long"], value="medium", label="Length")
            go = gr.Button("Summarize", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="Summary", lines=10, show_copy_button=True)
            info = gr.Markdown()
    go.click(run_pipeline, [url_in, text_in, target_in, mode_in, length_in], [out, info])

if __name__ == "__main__":
    demo.queue().launch(share=os.getenv("GRADIO_SHARE", "0") == "1")
