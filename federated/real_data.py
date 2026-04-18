
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
import os
from PIL import Image
from torchvision import transforms
import sys 

# ====================================================================
# 1. DJANGO SETUP & IMPORTS
# ====================================================================

# Django environment is expected to be loaded by manage.py or WSGI

from django.conf import settings 
from cervical.models import PatientRecord # Your model import

# ====================================================================
# 2. DATASET AND CLIENT IMPLEMENTATION (PyTorch/Flower Standard)
# ====================================================================

# Define the transformation pipeline (adjust as needed for your model)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CervicalRecordDataset(Dataset):
    """PyTorch Dataset for CervicalRecord data."""
    def __init__(self, records: List[Dict], transform=None):
        self.transform = transform
        self.media_root = settings.MEDIA_ROOT 
        
        # Filter records to only include those with valid existing files
        valid_records = []
        for rec in records:
            if not rec.get('image'):
                continue
            
            image_path = os.path.join(self.media_root, rec['image'])
            if os.path.isfile(image_path):
                valid_records.append(rec)
            else:
                print(f"Dataset warning: Image not found at {image_path}, skipping.")
                
        self.records = valid_records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        record = self.records[idx]
        
        # 'image' value from the DB is the path RELATIVE to MEDIA_ROOT
        image_relative_path = record['image'] 

        # Construct the ABSOLUTE file system path
        image_path = os.path.join(self.media_root, image_relative_path)
        
        # Load the image and apply transforms
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # FIX: Use 'image_label' instead of 'label'
            label = record['image_label'] 
            
            # Convert 'High' -> 1, 'Low' -> 0 if it's a string, or ensure it's mapped correctly.
            # Assuming image_label is stored as 'High'/'Low' strings based on previous code usage
            # rec.image_label = "High" if ... else "Low"
            if isinstance(label, str):
                target = 1 if label == 'High' else 0
            else:
                 # Fallback if it's already an int or something else
                target = 1 if label == 1 or label == '1' else 0

            return image, torch.tensor(target, dtype=torch.long)
        
        except Exception as e:
            # Should not happen often if filtered, but safety net
            print(f"Error loading image {image_path}: {e}")
            # If we still fail, we must SKIP or fail hard. 
            # Returning -1 crashes FL. Let's return a valid label 0 (Low) but zero-image.
            # Ideally we shouldn't reach here.
            return torch.zeros((3, 224, 224)), torch.tensor(0, dtype=torch.long)

def load_local_data(client_id: int = None):
    """
    Load data for a specific client_id (patient) or all data if client_id is None.
    If client_id passed, filters by patient__pk=client_id.
    """
    
    if client_id:
         client_records_query = PatientRecord.objects.filter(
            patient__pk=client_id 
        ).values('image', 'image_label')
    else:
        # Fallback: maybe load all? Or just return empty if strict.
        # For this use case, we probably want the specific patient's data.
        # But if we are simulating one client per patient, this is fine.
        client_records_query = PatientRecord.objects.all().values('image', 'image_label')

    client_records = list(client_records_query)
    
    if not client_records:
        # If no records, we can return empty loaders or handle gracefully
        print(f"No records found for client ID: {client_id}")
        return None, None

    # For a genuine FL client, we might use all data for training?
    # Or split. Let's keep the split for now.
    split_idx = int(len(client_records) * 0.8)
    # Ensure at least one sample in train if possible
    if split_idx == 0 and len(client_records) > 0:
        split_idx = len(client_records) 

    train_data = client_records[:split_idx]
    test_data = client_records[split_idx:]
    
    # If test data is empty, use train data for eval (just to avoid crashes in demo)
    if not test_data:
        test_data = train_data

    trainset = CervicalRecordDataset(train_data, transform=transform)
    testset = CervicalRecordDataset(test_data, transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    valloader = DataLoader(testset, batch_size=32)
    
    print(f"Client {client_id}: Loaded {len(train_data)} train, {len(test_data)} test records.")
    return trainloader, valloader
