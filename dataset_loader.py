# dataset_loader.py
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
MODEL_NAME = "microsoft/codebert-base"

class CodeDataset(Dataset):
    def __init__(self, csv_path, max_len=256):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.max_len = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        code = str(row['code'])
        label = int(row['label'])  # 0 = no vuln, 1 = vuln  (adapt if multiclass later)
        toks = self.tokenizer(code, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item = {k: v.squeeze(0) for k,v in toks.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item
