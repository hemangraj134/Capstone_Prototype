import torch
import chromadb
import base64
from model import CodeBERTClassifier
from sentence_transformers import SentenceTransformer

# 1. Config & Paths
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "G:/Capstone_Prototype/capstone_prototype_008/cwe_classifier.pth"
DB_PATH = "G:/Capstone_Prototype/capstone_prototype_008/vector_vault"

# Reverend Mapping (Matches your dataset_loader.py)
CWE_MAP_REV = {v: k for k, v in {
    "CWE-79": 0, "CWE-119": 1, "CWE-125": 2, "CWE-416": 3,
    "CWE-787": 4, "CWE-476": 5, "CWE-190": 6, "CWE-22": 7,
    "CWE-20": 8, "CWE-78": 9, "CWE-89": 10, "CWE-94": 11,
    "CWE-352": 12, "CWE-434": 13, "CWE-862": 14, "CWE-200": 15,
    "CWE-287": 16, "CWE-306": 17, "CWE-502": 18, "CWE-77": 19
}.items()}

class CapstoneScanner:
    def __init__(self):
        print(f"📡 Initializing Scanner on {DEVICE}...")
        
        # Load the "Detective" (Federated Brain)
        self.classifier = CodeBERTClassifier(num_labels=20).to(DEVICE)
        self.classifier.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        self.classifier.eval()

        # Load the "Librarian" (Embedder)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Connect to the "Vault" (Vector Database)
        self.db_client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.db_client.get_collection(name="circl_patches")

    def analyze_and_remediate(self, code_snippet):
        # --- PHASE 1: DETECTION ---
        inputs = self.classifier.tokenizer(code_snippet, return_tensors="pt", 
                                          truncation=True, padding="max_length", 
                                          max_length=256).to(DEVICE)
        
        with torch.no_grad():
            logits = self.classifier(inputs['input_ids'], inputs['attention_mask'])
            pred_idx = torch.argmax(logits, dim=1).item()
            cwe_id = CWE_MAP_REV.get(pred_idx, "Unknown")

        # --- PHASE 2: RETRIEVAL ---
        # Search the vault for the most similar vulnerability context
        query_vec = self.embedder.encode([code_snippet]).tolist()
        
        # We query the Top 1 match
        results = self.collection.query(
            query_embeddings=query_vec,
            n_results=1
        )

        if not results['metadatas'][0]:
            return f"🚨 Detected {cwe_id}, but no direct patch found in Vault."

        match = results['metadatas'][0][0]
        
        # --- PHASE 3: DECODING ---
        # Extract and decode the patch
        patch_b64 = match.get('patch_b64', '')
        decoded_patch = base64.b64decode(patch_b64).decode('utf-8', errors='ignore')

        return {
            "cwe": cwe_id,
            "title": match['title'],
            "source": match['id'],
            "fix": decoded_patch
        }

if __name__ == "__main__":
    scanner = CapstoneScanner()
    
    # Intentionally vulnerable C++ test code (Buffer Overflow pattern)
    test_code = """
    void handle_input(char *user_data) {
        char buffer[16];
        strcpy(buffer, user_data); // Vulnerable: No bounds check
    }
    """
    
    report = scanner.analyze_and_remediate(test_code)
    
    print("\n" + "="*60)
    print(f"🚩 VULNERABILITY DETECTED: {report['cwe']}")
    print(f"📄 CONTEXT: {report['title']}")
    print(f"🔗 SOURCE: {report['source']}")
    print("-" * 60)
    print("🛠️ PROPOSED HUMAN-VERIFIED FIX:")
    print(report['fix'])
    print("="*60)