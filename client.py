# client.py
import flwr as fl
import torch
from torch.utils.data import DataLoader
from model import CodeBERTClassifier
from dataset_loader import CodeDataset
import torch.nn as nn
import torch.optim as optim
import gc

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # speeds up GPU performance for constant input shapes

class FLClient(fl.client.NumPyClient):
    def __init__(self, train_csv, val_csv, epochs=1, batch_size=2, lr=1e-5):
        print(f"🧠 Initializing client on {DEVICE}...")
        self.model = CodeBERTClassifier()
        self.model.to(DEVICE)

        self.epochs = epochs
        self.batch_size = batch_size
        self.opt = optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        print(f"📂 Loading dataset: {train_csv}, {val_csv}")
        self.train_loader = DataLoader(CodeDataset(train_csv), batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(CodeDataset(val_csv), batch_size=batch_size)
        print(f"✅ Dataset loaded. Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")

    # -----------------------------
    # Federated Learning Interface
    # -----------------------------
    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, new in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new, device=DEVICE, dtype=p.dtype)

    # -----------------------------
    # Local Training
    # -----------------------------
    def fit(self, parameters, config):
        print("🚀 Starting local training round...")
        self.set_parameters(parameters)
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            for i, batch in enumerate(self.train_loader):
                input_ids = batch["input_ids"].to(DEVICE)
                attn = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                logits = self.model(input_ids, attn)
                loss = self.criterion(logits, labels)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                total_loss += loss.item()
                if (i + 1) % 100 == 0:
                    print(f"  [Epoch {epoch+1}] Batch {i+1}/{len(self.train_loader)} | Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(self.train_loader)
            print(f"✅ Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")

            gc.collect()
            torch.cuda.empty_cache()

        print("📤 Sending updated weights back to server.")
        return self.get_parameters(), len(self.train_loader.dataset), {}

    # -----------------------------
    # Local Evaluation
    # -----------------------------
    def evaluate(self, parameters, config):
        try:
            print("🧪 Evaluating global model on validation set...")
            self.set_parameters(parameters)
            self.model.eval()

            correct = 0
            total = 0
            loss_sum = 0.0

            with torch.no_grad():
                for batch in self.val_loader:
                    input_ids = batch["input_ids"].to(DEVICE)
                    attn = batch["attention_mask"].to(DEVICE)
                    labels = batch["labels"].to(DEVICE)

                    logits = self.model(input_ids, attn)
                    loss = self.criterion(logits, labels)

                    loss_sum += loss.item() * labels.size(0)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            avg_loss = loss_sum / total if total > 0 else 0.0
            accuracy = correct / total if total > 0 else 0.0
            print(f"📊 Validation Accuracy: {accuracy:.4f} | Loss: {avg_loss:.4f}")

            # Flower requires this exact tuple format
            return float(avg_loss), int(total), {"accuracy": float(accuracy)}

        except Exception as e:
            print(f"❗ Evaluation error: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            # Return fallback (so Flower doesn’t crash)
            return 1.0, 0, {"accuracy": 0.0}

# -----------------------------
# Client Launcher
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to client train.csv")
    parser.add_argument("--val", required=True, help="Path to client val.csv")
    parser.add_argument("--server", default="127.0.0.1:8080", help="Server address")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    client = FLClient(args.train, args.val, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    print(f"🤝 Connecting client to server at {args.server}")
    fl.client.start_numpy_client(server_address=args.server, client=client)
