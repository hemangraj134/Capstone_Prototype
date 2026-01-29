import subprocess
import time
import os

# Path to the python executable INSIDE your G: drive .venv
# This ensures clients find 'flwr', 'torch', etc.
venv_python = os.path.join(".venv", "Scripts", "python.exe")

clients = [
    {"train": "clients/client_0/train.csv", "val": "clients/client_0/val.csv"},
    {"train": "clients/client_1/train.csv", "val": "clients/client_1/val.csv"},
]

processes = []

print("🚀 Launching Federated Learning Clients (using .venv)...")

for i, paths in enumerate(clients):
    cmd = [
        venv_python, "client.py", # <-- Updated to use venv_python
        "--train", paths["train"],
        "--val", paths["val"],
        "--epochs", "1",
        "--batch_size", "2"
    ]
    print(f"Starting Client {i}...")
    p = subprocess.Popen(cmd)
    processes.append(p)
    time.sleep(2)

print(f"✅ {len(processes)} clients are now running.")

for p in processes:
    p.wait()