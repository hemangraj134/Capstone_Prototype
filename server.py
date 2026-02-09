import flwr as fl
import torch
import os
from model import CodeBERTClassifier

class RAGPersistenceStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        agg_params, agg_metrics = super().aggregate_fit(server_round, results, failures)

        if agg_params is not None and server_round == 5:
            print(f"💾 SAVING MULTI-CLASS DETECTOR...")
            ndarrays = fl.common.parameters_to_ndarrays(agg_params)
            
            # Initialize model with 20 labels to match training
            model = CodeBERTClassifier(num_labels=20)
            state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), ndarrays)}
            
            save_path = "G:/Capstone_Prototype/capstone_prototype_008/cwe_classifier.pth"
            torch.save(state_dict, save_path)
            print(f"✅ Classifier saved: {save_path}")
            
        return agg_params, agg_metrics

if __name__ == "__main__":
    strategy = RAGPersistenceStrategy(min_fit_clients=2, min_available_clients=2)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5)
    )