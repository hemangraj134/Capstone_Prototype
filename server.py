# server.py
import flwr as fl

if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,        # all clients participate each round
        min_fit_clients=2,       # minimum number of clients required
        min_available_clients=2, # both clients must be available
    )

    print("🚀 Starting Flower server with FedAvg strategy...")

    # ✅ Flower v1.8+ requires `fl.server.start_server` with keyword `config=fl.server.ServerConfig`
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5)  # new correct API
    )
