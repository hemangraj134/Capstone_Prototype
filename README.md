

---

# 🛡️ Federated RAG: Decentralized CWE Detection & Remediation

This repository contains a high-performance, privacy-preserving prototype for identifying and fixing C/C++ vulnerabilities. By integrating **Federated Learning (FL)** for detection and **Retrieval-Augmented Generation (RAG)** for remediation, this system provides human-verified security patches without ever exposing proprietary source code to a central server.

---

## 🚀 Project Overview

Traditional AI security tools often compromise data privacy by requiring code uploads to the cloud, or they "hallucinate" insecure fixes. This project implements a **"Detective & Librarian"** hybrid architecture:

* **The Detective:** A Federated **CodeBERT-LoRA** model that classifies vulnerability patterns (CWEs) locally.
* **The Librarian:** A RAG-driven engine that retrieves **Grounded Truth** patches from the human-verified **CIRCL** knowledge base.

### 💎 Key Innovations

* **Privacy-Preserving:** Built on the **Flower (flwr)** framework; raw source code never leaves the local client environment.
* **Parameter-Efficient Fine-Tuning (PEFT):** Utilizes **LoRA (Low-Rank Adaptation)** to train a 110M parameter transformer on consumer-grade hardware (**NVIDIA RTX 4050**).
* **Zero-Hallucination Remediation:** Unlike generative models, our system retrieves **verified Git patches** from the **CIRCL vault** to ensure patches are technically sound and expert-signed.

---

## 🏗️ System Architecture

### 1. Federated Learning Core (`server.py` & `client.py`)

* **Protocol:** Uses **FedAvg (Federated Averaging)** to synchronize local weight updates into a global model.
* **Model:** **CodeBERT-LoRA**—an encoder specialized for programming languages, optimized to update only  of its parameters, significantly reducing VRAM requirements.

### 2. The Remediation Vault (`index_vault.py` & `vector_vault/`)

* **Knowledge Base:** 35,000+ real-world patches from the **CIRCL** (Computer Incident Response Center Luxembourg) dataset.
* **Vector Engine:** **ChromaDB** Persistent Vector Store using `all-MiniLM-L6-v2` embeddings for 384-dimensional semantic retrieval.

### 3. The Scanner (`scanner.py`)

The integration layer where the **Classifier** identifies a vulnerability and the **RAG Engine** maps it to the closest historical fix.

---

## 🛠️ Tech Stack

| Component | Technology | Technical Detail |
| --- | --- | --- |
| **FL Framework** | **Flower (flwr)** | gRPC-based communication rounds |
| **Detection Engine** | **CodeBERT** | LoRA-injected Transformer (PEFT) |
| **Vector DB** | **ChromaDB** | Persistent local vector storage |
| **Compute Ops** | **PyTorch / CUDA** | Optimized for 6GB VRAM |
| **Datasets** | **DiverseVul & CIRCL** | 150k+ vulnerabilities / 35k+ patches |

---

## 🚦 Execution Workflow

### 1. Initialize the Knowledge Vault

Ensure the **CIRCL** dataset is in the root directory, then run:

```bash
python index_vault.py

```

### 2. Execute Federated Training

**Terminal 1 (Server):**

```bash
python server.py

```

**Terminal 2 (Client Launcher):**

```bash
python start_client.py

```

### 3. Run the Scanner (Remediation Loop)

Once `cwe_classifier.pth` is generated, run the final inference:

```bash
python scanner.py

```

---

## 📊 Technical Metrics (Current Status)

* **Convergence:** Successfully reduced Global Loss from **1.8598** to **1.8439** in 5 rounds.
* **Hardware Ops:** Benchmarked at **~2.5GB VRAM** usage (RTX 4050).
* **Retrieval:** Successfully mapped complex Kernel-level flaws to **CVE-2015-1805**.

---

## 🤝 Data & Artifact Management

*Large binary files are excluded from this repo via `.gitignore`. Members must follow this sync protocol:*

1. **Datasets:** Download `diversevul_20230702.json` and `circl_patches.csv` manually to the root.
2. **Weights:** If you wish to bypass training, download the latest `cwe_classifier.pth` from the shared Team Drive and place it in the root directory.

---
