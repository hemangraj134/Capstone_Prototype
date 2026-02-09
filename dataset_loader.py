# dataset_loader.py
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "microsoft/codebert-base"

# This dictionary maps text categories to numerical indices (0-19)
CWE_MAP = {
    "CWE-79": 0, "CWE-119": 1, "CWE-125": 2, "CWE-416": 3,
    "CWE-787": 4, "CWE-476": 5, "CWE-190": 6, "CWE-22": 7,
    "CWE-20": 8, "CWE-78": 9, "CWE-89": 10, "CWE-94": 11,
    "CWE-352": 12, "CWE-434": 13, "CWE-862": 14, "CWE-200": 15,
    "CWE-287": 16, "CWE-306": 17, "CWE-502": 18, "CWE-77": 19
}

class CodeDataset(Dataset):
    def __init__(self, csv_path, max_len=256):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.max_len = max_len

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        code = str(row['code'])
        
        # Technical Logic: Convert 'CWE-XXX' string to an integer ID
        # If the CWE isn't in our Top 20, it defaults to 0
        cwe_string = str(row['cwe'])
        label = CWE_MAP.get(cwe_string, 0) 

        toks = self.tokenizer(
            code, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_len, 
            return_tensors='pt'
        )
        
        item = {k: v.squeeze(0) for k,v in toks.items()}
        # Labels must be LongTensor for CrossEntropyLoss
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item