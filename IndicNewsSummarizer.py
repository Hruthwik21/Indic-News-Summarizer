'''!pip install transformers==4.38.2 curl_cffi beautifulsoup4 sentencepiece langdetect gradio -q'''
'''!pip install git+https://github.com/VarunGumma/IndicTransToolkit.git -q'''
'''
!pip install huggingface_hub -q
from huggingface_hub import login

login("Hugging face token")'''

from curl_cffi import requests as c_requests
from bs4 import BeautifulSoup
from langdetect import detect
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit import IndicProcessor
import gradio as gr

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Indic-to-English Engine
print("Loading Indic -> English Engine...")
in_en_name = "ai4bharat/indictrans2-indic-en-dist-200M"
in_en_tokenizer = AutoTokenizer.from_pretrained(in_en_name, trust_remote_code=True)
in_en_model = AutoModelForSeq2SeqLM.from_pretrained(in_en_name, trust_remote_code=True).to(device)

# 2. Load the Factual Summarizer (The Pivot)
print("Loading Factual Summarizer Engine...")
sum_model_name = "facebook/bart-large-cnn"
sum_tokenizer = AutoTokenizer.from_pretrained(sum_model_name)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_model_name).to(device)

# 3. Load English-to-Indic Engine
print("Loading English -> Indic Engine...")
en_in_name = "ai4bharat/indictrans2-en-indic-dist-200M"
en_in_tokenizer = AutoTokenizer.from_pretrained(en_in_name, trust_remote_code=True)
en_in_model = AutoModelForSeq2SeqLM.from_pretrained(en_in_name, trust_remote_code=True).to(device)

# Initialize the toolkit processor
ip = IndicProcessor(inference=True)

lang_map = {
    "Hindi": "hin_Deva", "Telugu": "tel_Telu", "Tamil": "tam_Taml",
    "Kannada": "kan_Knda", "Malayalam": "mal_Mlym", "Bengali": "ben_Beng",
    "Marathi": "mar_Deva", "Gujarati": "guj_Gujr", "Punjabi": "pan_Guru",
    "Odia": "ory_Orya", "Assamese": "asm_Beng", "English": "eng_Latn"
}

source_lang_map = {
    "en": "eng_Latn", "hi": "hin_Deva", "te": "tel_Telu", "ta": "tam_Taml",
    "ml": "mal_Mlym", "mr": "mar_Deva", "bn": "ben_Beng", "gu": "guj_Gujr",
    "pa": "pan_Guru", "kn": "kan_Knda", "or": "ory_Orya"
}

def extract_text_stealth(url):
    try:
        response = c_requests.get(url, impersonate="chrome110", timeout=15)
        if response.status_code != 200: return ""
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
    except:
        return ""

def process_news_flawless(url, target_lang_name):
    try:
        # A. Extract Text
        raw_text = extract_text_stealth(url)
        if not raw_text or len(raw_text) < 300:
            return "Error: Could not extract text. The firewall blocked us or the article is too short."
        text_chunk = raw_text[:3500]

        # B. Detect Source Language
        try:
            detected_code = detect(text_chunk)
            src_indic_code = source_lang_map.get(detected_code, "eng_Latn")
        except:
            src_indic_code = "eng_Latn"

        # C. Normalize to English (IndicTrans2)
        if src_indic_code != "eng_Latn":
            batch = ip.preprocess_batch([text_chunk], src_lang=src_indic_code, tgt_lang="eng_Latn")
            inputs = in_en_tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(device)
            outputs = in_en_model.generate(**inputs, max_length=1024, num_beams=4)
            decoded = in_en_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            english_text = ip.postprocess_batch(decoded, lang="eng_Latn")[0]
        else:
            english_text = text_chunk

        # D. Factual Summarization (BART)
        sum_inputs = sum_tokenizer(english_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
        sum_tokens = sum_model.generate(
            **sum_inputs,
            max_length=150,
            min_length=40,
            num_beams=4,
            length_penalty=2.0
        )
        english_summary = sum_tokenizer.decode(sum_tokens[0], skip_special_tokens=True)

        tgt_indic_code = lang_map[target_lang_name]
        if tgt_indic_code == "eng_Latn":
            return english_summary

        # E. Localize to Target Language (IndicTrans2)
        batch = ip.preprocess_batch([english_summary], src_lang="eng_Latn", tgt_lang=tgt_indic_code)
        inputs = en_in_tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(device)
        outputs = en_in_model.generate(**inputs, max_length=250, num_beams=5)
        decoded = en_in_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        final_summary = ip.postprocess_batch(decoded, lang=tgt_indic_code)[0]

        return final_summary

    except Exception as e:
        return f"System Error: {str(e)}"

# UI Launch
gr.Interface(
    fn=process_news_flawless,
    inputs=[
        gr.Textbox(label="News Article URL", placeholder="Paste any English or Indian news link..."),
        gr.Dropdown(list(lang_map.keys()), label="Translate Summary To", value="Telugu")
    ],
    outputs=gr.Textbox(label="Flawless Indic AI Summary"),
    title="IndicTrans2 Master News Summarizer",
    description="Uses India's state-of-the-art translation models (AI4Bharat) and an English Pivot to provide culturally and grammatically flawless summaries."
).launch(share=True)
