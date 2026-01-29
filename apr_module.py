# apr_module.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

APR_MODEL = "Salesforce/codet5-small"  # or t5-small adapted for code

tokenizer = AutoTokenizer.from_pretrained(APR_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(APR_MODEL).to("cuda" if torch.cuda.is_available() else "cpu")

def suggest_fix(code_snippet, max_length=128):
    prompt = "fix the vulnerability:\n" + code_snippet
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    out = model.generate(**tokens, max_length=max_length, num_beams=3)
    return tokenizer.decode(out[0], skip_special_tokens=True)
