## README.md

# Software Vulnerability Detection and Mitigation using Federated Learning

This repository contains the prototype for a Capstone project focused on identifying and mitigating security vulnerabilities in C/C++ source code. By leveraging **Federated Learning (FL)** and the **DiverseVul** dataset, this framework allows for the collective training of a robust detection model without requiring participants to share their private source code.

---

## 🚀 Project Overview

Traditional vulnerability detection requires centralized datasets, which is often impossible for organizations with sensitive proprietary code. This project solves that by using the **Flower (flwr)** framework to decentralize the training process.

### Key Features

* **Privacy-Preserving:** Training happens locally on client machines; only model updates (weights) are sent to the server.
* **C/C++ Focused:** Specifically tuned to handle the complexities of C/C++ memory management and syntax using the DiverseVul dataset.
* **Scalable:** Designed to support multiple clients using Federated Averaging (**FedAvg**).

---

## 🏗️ System Architecture

The system consists of two primary components:

### 1. The Server (`server.py`)

The central coordinator responsible for:

* Initializing the global model.
* Selecting available clients for training rounds.
* Aggregating local weights using the **FedAvg** strategy.
* Evaluating the global model's performance on a validation set.

### 2. The Clients (`client.py`)

Individual nodes that:

* Host their own slice of the **DiverseVul** dataset.
* Perform local training using deep learning models (e.g., GRU, LSTM, or GNN).
* Transmit only the resulting weight changes back to the server.

---

## 📊 Dataset: DiverseVul

The project utilizes **DiverseVul**, a comprehensive dataset of C/C++ vulnerabilities.

* **Diversity:** Covers 150+ CWE types (Common Weakness Enumeration).
* **Source:** Real-world vulnerabilities from open-source projects.
* **Format:** Typically pre-processed into tokenized sequences or graph representations for model ingestion.

---

## 🛠️ Tech Stack

| Component | Technology |
| --- | --- |
| **FL Framework** | Flower (`flwr`) |
| **Deep Learning** | PyTorch / TensorFlow |
| **Language** | Python 3.x |
| **Analysis Target** | C / C++ |
| **Dataset** | DiverseVul |

---

## 🚦 Getting Started

### Prerequisites

* Python 3.8+
* `pip install flwr torch numpy pandas`

### Execution

1. **Start the Server:**
```bash
python server.py

```


2. **Start the Clients (in separate terminals):**
```bash
python client.py --client_id 1
python client.py --client_id 2

```



---
