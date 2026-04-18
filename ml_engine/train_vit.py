import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from PIL import Image

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT_DIR = os.path.join(BASE_DIR, 'data')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "image_vit.pth")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# FIX: Reduced batch size to 16 is good. If you get "CUDA Out of Memory", drop this to 8.
batch_size = 16 
img_size = 224
epochs = 20
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard ImageNet Normalization (Correct for ViT too)
normalize_params = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15), # Increased slightly for more robustness
    transforms.ToTensor(),
    transforms.Normalize(*normalize_params)
])

val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(*normalize_params)
])

# --- Model Building ---
def build_model(num_classes):
    # Compatibility: Handle both old and new torchvision versions
    try:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    except:
        model = models.vit_b_16(pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # FIX: Robust head replacement
    # We check if 'heads' exists and has a 'head' attribute (standard in torchvision ViT)
    if hasattr(model, 'heads') and hasattr(model.heads, 'head'):
        in_features = model.heads.head.in_features
    else:
        # Fallback: Assume standard ViT hidden dimension
        in_features = 768 
    
    # Overwrite the head with your specific class count
    model.heads = nn.Linear(in_features, num_classes)
    return model

# --- Training Loop ---
def train_model():
    try:
        train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT_DIR, "train"), transform=train_transform)
        val_ds   = datasets.ImageFolder(os.path.join(DATA_ROOT_DIR, "val"), transform=val_transform)
    except Exception as e:
        print(f"❌ Dataset Error: {e}")
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2) # Added workers for speed
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"🚀 Training ViT on {num_classes} classes: {class_names}")

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Optimize ONLY the new head (gradients are on for this part only)
    optimizer = optim.AdamW(model.heads.parameters(), lr=lr) # FIX: AdamW is better for Transformers

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(outputs.argmax(dim=1) == labels).item()

        epoch_loss = running_loss / len(train_ds)
        epoch_acc = running_corrects / len(train_ds)

        # Validation Phase
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # FIX: Added .item() here to prevent GPU memory leak over time
                val_corrects += torch.sum(outputs.argmax(dim=1) == labels).item()

        val_acc = val_corrects / len(val_ds)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": class_names
            }, MODEL_PATH)
            print(f"  💾 Model Saved (Acc: {best_acc:.4f})")

    print(f"\n✅ Training Complete. Best Accuracy: {best_acc:.4f}")

# --- Prediction ---
def predict_image(image_path):
    if not os.path.exists(MODEL_PATH):
        return 0.0, "Model Not Found"

    try:
        # Load checkpoint
        ckpt = torch.load(MODEL_PATH, map_location=device)
        class_names = ckpt.get("classes", [])
        
        # FIX: Crash prevention if classes are missing
        if not class_names:
            return 0.0, "Error: No classes found in checkpoint"

        # Rebuild model with correct number of classes
        model = build_model(len(class_names)).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        # Process Image
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(*normalize_params)
        ])
        
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        max_idx = probs.argmax()
        return float(probs[max_idx]), class_names[max_idx]

    except Exception as e:
        return 0.0, f"Error: {str(e)}"

if __name__ == "__main__":
    train_model()