import torch
import warnings
# Suppress MatplotlibDeprecationWarning which appears in torchcam
try:
    from matplotlib import MatplotlibDeprecationWarning
    warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", message=".*get_cmap function was deprecated.*")
from torchvision import transforms, models as tvmodels
from PIL import Image
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
import numpy as np
import os

# --- DYNAMIC PATHS ---
# Get the base directory dynamically (the directory containing this file is 'src')
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)  # cervical_multimodal directory

# Path where the trained PyTorch model is located (assuming it's in a 'models' folder)
MODEL_PATH = os.path.join(BASE_DIR, "models", "image_vit.pth")
GRADCAM_OUTPUT_DIR = os.path.join(BASE_DIR, "cervical", "static", "cervical", "uploads", "gradcam")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Load Model ---
def load_model():
    """Load the model and configure the final layer based on saved class count."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Image model not found at {MODEL_PATH}")
    
    ckpt = torch.load(MODEL_PATH, map_location=device)
    
    # Handle both direct state_dict saves and full checkpoint dictionaries
    if isinstance(ckpt, dict):
        saved_classes = ckpt.get('classes', None)
        state_dict = ckpt.get('state_dict', ckpt)
    else:
        # If ckpt is directly the state_dict
        saved_classes = None
        state_dict = ckpt

    # Determine number of classes
    if saved_classes is None:
        print("⚠️ Warning: Class names not found in checkpoint. Inferring from model structure.")
        # Try to infer from heads.head.weight shape in state_dict (ViT specific)
        if 'heads.head.weight' in state_dict:
            num_classes = state_dict['heads.head.weight'].shape[0]
        elif 'head.weight' in state_dict: # Some ViT variants
             num_classes = state_dict['head.weight'].shape[0]
        else:
            print("⚠️ Warning: Could not infer number of classes. Assuming 5 classes.")
            num_classes = 5
    else:
        num_classes = len(saved_classes)

    try:
        model = tvmodels.vit_b_16(weights=None)
    except:
        model = tvmodels.vit_b_16(pretrained=False)

    # ViT Head Replacement
    if hasattr(model, 'heads') and hasattr(model.heads, 'head'):
        in_features = model.heads.head.in_features
        model.heads = torch.nn.Linear(in_features, num_classes)
    else:
        model.heads = torch.nn.Linear(768, num_classes)
    
    # Load state dict, handling potential prefix issues
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"⚠️ Strict loading failed: {e}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device).eval()
    return model

class ViTGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def __call__(self, class_idx, scores):
        # Zero gradients
        self.model.zero_grad()
        
        # Backpropagate
        loss = scores[0, class_idx]
        loss.backward(retain_graph=True)
        
        # Get activations and gradients
        # Shape: (B, 197, 768)
        act = self.activations
        grad = self.gradients
        
        # Handle ViT shapes: Remove CLS token (index 0) if present (197 tokens)
        if act.shape[1] == 197:
            act = act[:, 1:, :] 
            grad = grad[:, 1:, :]
            
        # Reshape to (B, C, H, W) -> (1, 768, 14, 14)
        # Assuming square grid
        num_tokens = act.shape[1]
        grid_size = int(np.sqrt(num_tokens))
        embed_dim = act.shape[2]
        
        act = act.permute(0, 2, 1).reshape(1, embed_dim, grid_size, grid_size)
        grad = grad.permute(0, 2, 1).reshape(1, embed_dim, grid_size, grid_size)
        
        # Compute weights (Global Average Pooling of gradients)
        weights = grad.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = (weights * act).sum(dim=1, keepdim=True)
        
        # ReLU
        cam = torch.nn.functional.relu(cam)
        
        # Clean up hooks
        # for h in self.hooks: h.remove() # Keep hooks for reuse or managing lifecycle? 
        # For this script, usage implies one-off. But better keep it simple.
        
        return cam
        
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

# --- Generate Grad-CAM ---
def generate_gradcam(img_input, filename_base):
    """
    Generate Grad-CAM overlay and save to the predefined GRADCAM_OUTPUT_DIR.
    Args:
        img_input (str or PIL.Image): Image path or image object
        filename_base (str): Base name for the output file
    Returns:
        str: Path to the saved Grad-CAM image
    """
    try:
        if isinstance(img_input, str):
            img = Image.open(img_input).convert("RGB")
        else:
            img = img_input

        model = load_model()
        
        # USE CUSTOM ViT EXTRACTOR
        # model.encoder.layers[-1].ln_1 is a common choice for ViT-B/16
        cam_extractor = ViTGradCAM(model, target_layer=model.encoder.layers[-1].ln_1)

        input_tensor = transform(img).unsqueeze(0).to(device)
        output = model(input_tensor)
        pred_class = int(output.argmax(dim=1).item())

        # Returns directly the (B, 1, H, W) cam
        cam_tensor = cam_extractor(pred_class, output)
        
        # Squeeze to (H, W) -> (14, 14)
        activation = cam_tensor.squeeze().cpu().detach().numpy()

        # Normalize the activation map
        act = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)
        heatmap = (act * 255).astype('uint8')
        heat_pil = Image.fromarray(heatmap).convert('L')

        # ✅ Ensure input image is still a PIL image
        img_resized = img.resize((224, 224))
        overlay = overlay_mask(img_resized, heat_pil, alpha=0.5)

        # Save the image
        output_filename = f"gradcam_{filename_base}"
        save_path = os.path.join(GRADCAM_OUTPUT_DIR, output_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.imsave(save_path, overlay)

        print(f"✅ Grad-CAM visualization saved to: {save_path}")
        return save_path
    
    except Exception as e:
        print(f"❌ Error generating Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
        return ""

# --- Example usage ---
#if __name__ == '__main__':
    #TEST_IMAGE_PATH = r'C:\Users\user\OneDrive\Desktop\002_05.jpg'
    #if os.path.exists(TEST_IMAGE_PATH):
       # base_name = os.path.basename(TEST_IMAGE_PATH)
        #generate_gradcam(TEST_IMAGE_PATH, base_name)
    #else:
       # print(f"❌ Test image not found at {TEST_IMAGE_PATH}. Please provide a valid path to test.")
