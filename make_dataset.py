# make_dataset.py
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

DATA_PATH = "diversevul_20230702 (1).json"
OUT_DIR = "clients"

print("📂 Loading DiverseVul dataset (JSONL format)...")

records = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)

            # Get fields
            code = item.get("func", "")
            label = item.get("target", None)
            cwe_list = item.get("cwe", [])

            # Skip if missing essentials
            if not code or label is None:
                continue

            # Clean up and record
            cwe = cwe_list[0] if cwe_list else "Unknown"
            records.append({"code": code, "label": int(label), "cwe": cwe})

            if i < 2:
                print(f"Example #{i+1} → label={label}, cwe={cwe}")
        except Exception as e:
            continue

print(f"✅ Loaded {len(records)} valid samples")

if not records:
    print("⚠️ No valid samples found. Check JSON structure again.")
    exit()

df = pd.DataFrame(records)
# ⚡ Use only a smaller subset for prototype training (faster + memory safe)
df = df.sample(n=10000, random_state=42)
print(f"⚡ Using a smaller subset: {len(df)} samples for prototype testing")

print("Data preview:")
print(df.head())

# Create client splits (simulating federated learning)
os.makedirs(OUT_DIR, exist_ok=True)
clients = 2  # Increase if needed
splits = np.array_split(df.sample(frac=1, random_state=42), clients)

for i, split_df in enumerate(splits):
    train_df, val_df = train_test_split(split_df, test_size=0.2, random_state=42)
    client_path = os.path.join(OUT_DIR, f"client_{i}")
    os.makedirs(client_path, exist_ok=True)
    train_df.to_csv(os.path.join(client_path, "train.csv"), index=False)
    val_df.to_csv(os.path.join(client_path, "val.csv"), index=False)
    print(f"📁 Client {i}: {len(train_df)} train / {len(val_df)} val samples")

print("🎯 Data prepared successfully — ready for FL training.")
