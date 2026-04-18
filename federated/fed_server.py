import flwr as fl
from flwr.server.strategy import FedAvg
import numpy as np
import sys
from typing import Dict, Optional, Tuple, List
import os

# ====================================================================
# 1. DJANGO SETUP & IMPORTS
# ====================================================================
# ====================================================================
# 1. DJANGO SETUP & IMPORTS
# ====================================================================
# We assume setup_django_environment is imported and called here
try:
    from .setup_django import setup_django_environment
except ImportError:
    from setup_django import setup_django_environment

import django
from django.conf import settings

# Only setup if not already configured (avoids issues when imported in apps.py)
if not settings.configured:
    setup_django_environment()
from cervical.models import PatientRecord # Example model import

# ====================================================================
# 2. STRATEGY AND UTILS (Placeholders)
# ====================================================================

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    # Simple calculation for a weighted average (if needed)
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    if sum(examples) == 0:
        return {"accuracy": 0.0}
        
    return {"accuracy": sum(accuracies) / sum(examples)}

# Define the strategy
strategy = FedAvg(
    fraction_fit=1.0,  # Sample all clients in each round
    fraction_evaluate=1.0,
    min_fit_clients=1,  # Require at least one client to participate
    min_evaluate_clients=1, # Allow evaluation with just 1 client
    min_available_clients=1,
    evaluate_metrics_aggregation_fn=weighted_average,
)

# ====================================================================
# ====================================================================

def start_server():
    print("Starting Flower server...")
    
    # 🛑 FIX: Use a new, less common port (8088) for the server.
    SERVER_ADDRESS = "127.0.0.1:8095" 
    
    # Flower server configuration
    while True:
        try:
            print("🚀 Starting/Restarting FL server session...")
            fl.server.start_server(
                server_address=SERVER_ADDRESS,
                config=fl.server.ServerConfig(num_rounds=3),
                strategy=strategy,
            )
            print("✅ FL session finished.")
        except Exception as e:
            print(f"FAILED to start FL server (possibly port in use): {e}")
            import time
            time.sleep(5) # Wait before retry

if __name__ == "__main__":
    start_server()
