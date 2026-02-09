import base64
import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import tqdm

# 1. Configuration
DB_PATH = "G:/Capstone_Prototype/capstone_prototype_008/vector_vault"
COLLECTION_NAME = "circl_patches"

def build_vault():
    # 2. Initialize Vector DB (Disk-Persistent)
    # ChromaDB is chosen because it runs locally on your laptop without a server.
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # 3. Load the Embedding Model
    # 'all-MiniLM-L6-v2' is the "God-tier" model for laptops. 
    # It’s small (80MB) but highly accurate for semantic similarity.
    print("🧠 Loading Embedding Model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # 4. Stream CIRCL Dataset
    print("📡 Streaming CIRCL dataset for indexing...")
    ds = load_dataset("CIRCL/vulnerability-cwe-patch", split="train", streaming=True)

    batch_size = 100
    ids, documents, metadatas, embeddings = [], [], [], []

    for i, row in enumerate(ds):
        # Technical Filter: Skip entries without patches
        if not row['patches'] or not row['description']:
            continue
            
        # Prepare the metadata (The info we need to show the mentor)
        # We store the b64 patch so we can decode it during retrieval.
        meta = {
            "id": row['id'],
            "cwe": str(row['cwe']),
            "title": row['title'][:200], # Truncate for DB efficiency
            "patch_b64": row['patches'][0]['patch_text_b64'][:5000] # Limit patch size
        }

        # The 'document' is what the model "reads" to find similarities
        text_to_embed = f"CWE: {row['cwe']} | Title: {row['title']} | Description: {row['description']}"
        
        ids.append(f"id_{i}")
        documents.append(text_to_embed)
        metadatas.append(meta)
        
        # Periodic Batch Processing to save RAM
        if len(ids) >= batch_size:
            print(f"📥 Indexing batch up to row {i}...")
            vectors = embedder.encode(documents).tolist()
            collection.add(
                ids=ids,
                embeddings=vectors,
                metadatas=metadatas,
                documents=documents
            )
            # Clear buffers
            ids, documents, metadatas = [], [], []

    print(f"✨ SUCCESS: Vector Vault built at {DB_PATH}")

if __name__ == "__main__":
    build_vault()