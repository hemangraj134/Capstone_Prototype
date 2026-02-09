# client.py
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import gc
import argparse
from torch.utils.data import DataLoader
from model import CodeBERTClassifier
from dataset_loader import CodeDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True 

class FLClient(fl.client.NumPyClient):
    def __init__(self, train_csv, val_csv, epochs=1, batch_size=2, lr=1e-5):
        print(f"🧠 Initializing client on {DEVICE}...")
        # Directive: Multi-class (20 labels) for RAG mapping
        self.model = CodeBERTClassifier(num_labels=20).to(DEVICE)
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.opt = optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        print(f"📂 Loading: {train_csv}")
        self.train_loader = DataLoader(CodeDataset(train_csv), batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(CodeDataset(val_csv), batch_size=batch_size)

    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device=DEVICE) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(self.epochs):
            for batch in self.train_loader:
                input_ids, attn, labels = batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), batch["labels"].to(DEVICE).long()
                self.opt.zero_grad()
                loss = self.criterion(self.model(input_ids, attn), labels)
                loss.backward()
                self.opt.step()
        
        gc.collect()
        torch.cuda.empty_cache()
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids, attn, labels = batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), batch["labels"].to(DEVICE).long()
                logits = self.model(input_ids, attn)
                loss_sum += self.criterion(logits, labels).item() * labels.size(0)
                correct += (logits.argmax(dim=-1) == labels).sum().item()
                total += labels.size(0)
        
        return float(loss_sum/total), int(total), {"accuracy": float(correct/total)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # These match your start_client.py commands exactly
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--server", default="127.0.0.1:8080")
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    client = FLClient(args.train, args.val, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    fl.client.start_numpy_client(server_address=args.server, client=client)