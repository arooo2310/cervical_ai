
# cervical_multimodal/fedrated/fed_client.py

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models
from typing import Tuple, Dict, List
from collections import OrderedDict

# Import the real data loader
try:
    from .real_data import load_local_data
except ImportError:
    from real_data import load_local_data

# --- Model Definition (Vision Transformer) ---
def build_model(num_classes: int):
    """
    Builds a Vision Transformer (ViT) model.
    """
    try:
        # Try importing from newer torchvision versions
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    except:
        # Fallback for older versions
        model = models.vit_b_16(pretrained=True)

    # Freeze feature extractor parameters?
    # For FL, we might want to fine-tune some layers or just the head.
    # The reference implementation froze everything. 
    # Let's keep it consistent: freeze backbone, train head.
    for param in model.parameters():
        param.requires_grad = False

    # Robust head replacement
    if hasattr(model, 'heads') and hasattr(model.heads, 'head'):
        in_features = model.heads.head.in_features
    else:
        in_features = 768 # Standard ViT hidden dim
    
    # Overwrite the head with new class count
    model.heads = nn.Linear(in_features, num_classes)
    return model

# --- Flower Client Implementation ---
class CervicalClient(fl.client.NumPyClient):
    def __init__(self, model: nn.Module, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.CrossEntropyLoss()
        # Use AdamW for ViT as per best practices/reference
        self.optimizer = optim.AdamW(self.model.heads.parameters(), lr=1e-4)
        self.model.to(self.device)

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Get model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters from the server's aggregated weights."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model locally."""
        self.set_parameters(parameters)
        
        epochs = config.get("local_epochs", 1)
        self.model.train()
        
        for _ in range(epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate the model locally."""
        self.set_parameters(parameters)
        self.model.eval()
        
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.valloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Handle division by zero if valloader is empty 
        if total == 0:
            return 0.0, 0, {"accuracy": 0.0}

        avg_loss = loss / len(self.valloader)
        accuracy = correct / total
        
        return avg_loss, total, {"accuracy": accuracy}

def train_locally(client_id=None):
    """
    Function to trigger the FL client process.
    This is called by the Django view.
    """
    print(f"Starting FL client for client ID: {client_id}...")
    
    # 1. Load Data
    trainloader, valloader = load_local_data(client_id)
    
    if not trainloader:
        print("No data found, skipping FL participation.")
        return

    # 2. Load Model
    num_classes = 2 # High/Low
    model = build_model(num_classes)
    
    # 3. Create Client and Start Federated Learning
    client = CervicalClient(model, trainloader, valloader)
    
    # Connect to server
    try:
        # Use a timeout or run in a separate thread/process in production.
        # For now, we run it. Note: 'start_numpy_client' is blocking. 
        fl.client.start_client(
            server_address="127.0.0.1:8095", 
            client=client.to_client()
        )
    except Exception as e:
        import traceback
        error_msg = f"❌ Flower client failed to start/connect: {e}\n{traceback.format_exc()}"
        print(error_msg)
        # Log to file for troubleshooting
        with open("client_error.log", "w", encoding="utf-8") as f:
            f.write(error_msg)

if __name__ == "__main__":
    # Test run
    train_locally(client_id=1) 
