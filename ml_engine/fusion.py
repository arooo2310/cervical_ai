import joblib, torch, os
import numpy as np
from torchvision import transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMG_MODEL_PATH = os.path.join(MODELS_DIR, "image_vit.pth")
CLIN_MODEL_PATH = os.path.join(MODELS_DIR, "clinical_xgb.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "clinical_scaler.joblib")
ENC_PATH = os.path.join(MODELS_DIR, "clinical_ohe.joblib")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def fuse_probs(clin_prob, img_prob, w_clin=0.4, w_img=0.6):
    return float(w_clin*clin_prob + w_img*img_prob)


def load_image_model():
    ckpt = torch.load(IMG_MODEL_PATH, map_location=device)
    from torchvision import models as tvmodels
    model = tvmodels.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, ckpt.get('classes', 2))
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device).eval()
    return model

def predict_image_prob(img_path, model=None):
    model = model or load_image_model()
    img = Image.open(img_path).convert("RGB")
    x = img_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return float(probs.max()), int(probs.argmax())

def load_clinical_model():
    model = joblib.load(CLIN_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    try:
        enc = joblib.load(ENC_PATH)
    except Exception:
        enc = None
    return model, scaler, enc

def predict_clinical_prob(row_array, model=None, scaler=None, enc=None):
    if model is None:
        model, scaler, enc = load_clinical_model()
    arr_scaled = scaler.transform(row_array.reshape(1, -1))
    prob = model.predict_proba(arr_scaled)[0][1]
    return float(prob)


def run_multimodal_prediction(image_or_path, clinical_features_dict):
    import numpy as np
    row = np.array([float(v) for v in clinical_features_dict.values()])
    clin_model, scaler, enc = load_clinical_model()
    clin_prob = predict_clinical_prob(row, clin_model, scaler, enc)
    if isinstance(image_or_path, Image.Image):
        tmp_path = os.path.join(BASE_DIR, 'tmp', 'temp_image.jpg')
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        image_or_path.save(tmp_path)
        img_prob, _ = predict_image_prob(tmp_path)
        os.remove(tmp_path)
    else:
        img_prob, _ = predict_image_prob(image_or_path)
    fused = fuse_probs(clin_prob, img_prob)
    prediction_label = "Abnormal" if fused >= 0.50 else "Normal"
    return prediction_label
