import os
import joblib
import torch
import torch.nn as nn
from torchvision import models as tvmodels, transforms
from PIL import Image
import pandas as pd
from django.conf import settings # Import Django settings for static path prefix

from sklearn.impute import KNNImputer

# Use relative imports for modules within the 'src' package
from .gradcam import generate_gradcam
from .shap_explain import generate_shap
# Assuming you have a file named 'fusion.py' in the 'src' directory
# Assuming you have a file named 'fusion.py' in the 'src' directory
from .fusion import fuse_probs 

# ------------------- Paths & Device ------------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the common STATIC PATH PREFIX used by Django templates/models
STATIC_PATH_PREFIX = "cervical/uploads/"


# ------------------- Image Transform ------------------- #
from torchvision import transforms
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resolve models/ dynamically
SRC_DIR    = os.path.dirname(__file__)                 # .../cervical_multimodal/src
BASE_DIR   = os.path.dirname(SRC_DIR)                  # .../cervical_multimodal
MODELS_DIR = os.path.join(BASE_DIR, "models")          # .../cervical_multimodal/models


def _strip_prefix(state_dict, prefixes=("module.", "model.")):
    """Remove common prefixes added by DataParallel / Lightning."""
    if not isinstance(state_dict, dict):
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        new_sd[nk] = v
    return new_sd


def _infer_num_classes(saved_classes, state_dict):
    """Prefer explicit classes; otherwise infer from fc.weight shape; else default to 2."""
    if saved_classes is not None:
        try:
            return int(len(saved_classes))
        except Exception:
            pass

    # Try from classifier weight/bias
    for key in ("fc.weight", "classifier.weight"):
        if key in state_dict and state_dict[key].ndim == 2:
            return int(state_dict[key].shape[0])

    # Fallback
    return 2


def load_image_model():
    """
    Load a Vision Transformer (ViT) model for inference and return (model, class_names).
    Handles full model / state_dict / DataParallel / Lightning checkpoints.
    """
    candidates = [
        os.path.join(MODELS_DIR, "image_vit.pth"),
        os.path.join(getattr(settings, "BASE_DIR", BASE_DIR), "models", "image_vit.pth"),
    ]
    model_path = next((p for p in candidates if os.path.exists(p)), None)
    if model_path is None:
        raise FileNotFoundError(
            "Image model not found. Place 'image_vit.pth' in one of:\n- " + "\n- ".join(candidates)
        )

    ckpt = torch.load(model_path, map_location=device)
    class_names = ckpt.get("classes", []) if isinstance(ckpt, dict) else []

    if isinstance(ckpt, nn.Module):
        model = ckpt.to(device).eval()
        return model, class_names

    if not isinstance(ckpt, dict):
        raise RuntimeError("Unsupported checkpoint format for image_vit.pth")

    saved_classes = ckpt.get("classes", ckpt.get("class_names", None))
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = _strip_prefix(state_dict)

    if saved_classes:
        num_classes = len(saved_classes)
        class_names = saved_classes
    else:
        num_classes = _infer_num_classes(saved_classes, state_dict)
        print("⚠️ Warning: No class names in checkpoint. Assuming alphabetical order.")
        class_names = [f"Class_{i}" for i in range(num_classes)]

    # Build ViT model
    try:
        model = tvmodels.vit_b_16(weights=None)
    except:
        model = tvmodels.vit_b_16(pretrained=False) # Fallback for older versions

    # ViT Head Replacement
    if hasattr(model, 'heads') and hasattr(model.heads, 'head'):
        in_features = model.heads.head.in_features
        model.heads = nn.Linear(in_features, num_classes)
    else:
        # Fallback/Safety
        model.heads = nn.Linear(768, num_classes)

    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model, (saved_classes or class_names or [])

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def clinical_predict(features: dict, record_id: int | None = None):
    """
    Use saved artifacts (separate imputer, OHE, scaler, classifier).
    Build EXACT training schema: numeric -> imputer, categorical -> OHE, then concat.
    """
    import os, joblib, numpy as np, pandas as pd

    model       = joblib.load(os.path.join(MODELS_DIR, "clinical_xgb.joblib"))
    scaler      = joblib.load(os.path.join(MODELS_DIR, "clinical_scaler.joblib"))
    num_imp     = joblib.load(os.path.join(MODELS_DIR, "clinical_num_imputer.joblib"))
    try:
        ohe     = joblib.load(os.path.join(MODELS_DIR, "clinical_ohe.joblib"))
        cat_cols = list(getattr(ohe, "feature_names_in_", []))
        cat_choices = getattr(ohe, "categories_", None)
    except Exception:
        ohe = None
        cat_cols = []
        cat_choices = None
    feat_after  = joblib.load(os.path.join(MODELS_DIR, "clinical_feature_names.joblib"))

    # Raw training columns by branch
    if not hasattr(num_imp, "feature_names_in_"):
        raise RuntimeError("Numeric imputer lacks feature_names_in_. Save numeric raw columns during training.")
    num_cols = list(num_imp.feature_names_in_)

    # --- Synthesize a full raw row with training names ---
    row_num = {c: np.nan for c in num_cols}  # numeric defaults to NaN for imputation
    row_cat = {}
    if cat_choices is not None:
        for c, choices in zip(cat_cols, cat_choices):
            row_cat[c] = (choices[0] if len(choices) else "Unknown")

    ui = {
        "age": features.get("age", 0),
        "hpv_result": features.get("hpv_result", ""),
        "smoking": features.get("smoking", 0),
        "contraception": features.get("contraception", 0),
        "sexual_history": features.get("sexual_history", 0),
        "first_sexual_intercourse": features.get("first_sexual_intercourse", 0),
        "num_pregnancies": features.get("num_pregnancies", 0),
        "iud_years": features.get("iud_years", 0),
    }

    # Handle STDs:HPV if it is numeric
    hpv_ui = str(ui.get("hpv_result", "")).strip().lower()
    is_positive = hpv_ui in {"positive","pos","1","true","yes"}
    
    if "STDs:HPV" in row_num and is_positive:
        row_num["STDs:HPV"] = 1.0
        # If HPV is positive, STD count should be at least 1
        if "STDs: Number of diagnosis" in row_num:
            row_num["STDs: Number of diagnosis"] = 1.0
        if "STDs (number)" in row_num:
             row_num["STDs (number)"] = 1.0
             
    # Dx:HPV is typically categorical, handled in OHE loop below

    numeric_map = {
        "Age": "age",
        "Smokes (years)": "smoking",
        "Hormonal Contraceptives (years)": "contraception",
        "Number of sexual partners": "sexual_history",
        "First sexual intercourse": "first_sexual_intercourse",
        "Num of pregnancies": "num_pregnancies",
        "IUD (years)": "iud_years",
    }
    for train_col, ui_key in numeric_map.items():
        if train_col in row_num and ui_key in ui:
            row_num[train_col] = ui[ui_key]

    # Smokes
    s_years = float(ui.get("smoking") or 0)
    if "Smokes" in row_num:
        row_num["Smokes"] = 1.0 if s_years > 0 else 0.0
    if "Smokes" in row_cat:
        # If user entered smoking years > 0, assume Smokes='1.0', else if explicitly 0 => '0.0', else let default
        s_years = float(ui.get("smoking") or 0)
        # Try to match the format in cat_choices if possible
        val = "1.0" if s_years > 0 else "0.0"
        # Check against trained categories if we can find them
        if cat_choices is not None and "Smokes" in cat_cols:
            idx = cat_cols.index("Smokes")
            trained = set(map(str, cat_choices[idx]))
            if val not in trained and "1" in trained: val = "1"
            if val not in trained and "0" in trained and s_years == 0: val = "0"
            if val not in trained and "True" in trained: val = "True" if s_years > 0 else "False"
        row_cat["Smokes"] = val

    # Hormonal Contraceptives
    hc_years = float(ui.get("contraception") or 0)
    if "Hormonal Contraceptives" in row_num:
        row_num["Hormonal Contraceptives"] = 1.0 if hc_years > 0 else 0.0
    if "Hormonal Contraceptives" in row_cat:
        hc_years = float(ui.get("contraception") or 0)
        val = "1.0" if hc_years > 0 else "0.0"
        if cat_choices is not None and "Hormonal Contraceptives" in cat_cols:
            idx = cat_cols.index("Hormonal Contraceptives")
            trained = set(map(str, cat_choices[idx]))
            if val not in trained and "1" in trained: val = "1"
            if val not in trained and "0" in trained and hc_years == 0: val = "0"
            if val not in trained and "True" in trained: val = "True" if hc_years > 0 else "False"
        row_cat["Hormonal Contraceptives"] = val

    # IUD inferences
    iud_years = float(ui.get("iud_years") or 0)
    if "IUD" in row_num:
        row_num["IUD"] = 1.0 if iud_years > 0 else 0.0
    if "IUD" in row_cat:
        iud_years = float(ui.get("iud_years") or 0)
        val = "1.0" if iud_years > 0 else "0.0"
        if cat_choices is not None and "IUD" in cat_cols:
            idx = cat_cols.index("IUD")
            trained = set(map(str, cat_choices[idx]))
            if val not in trained and "1" in trained: val = "1"
            if val not in trained and "0" in trained and iud_years == 0: val = "0"
            if val not in trained and "True" in trained: val = "True" if iud_years > 0 else "False"
        row_cat["IUD"] = val

    # Normalize HPV and write to whichever categorical column exists
    hpv_ui = str(ui.get("hpv_result", "")).strip().lower()
    # Map to '1'/'0' as per OHE categories found in debug
    hpv_norm = "1" if hpv_ui in {"positive","pos","1","true","yes"} else \
               "0" if hpv_ui in {"negative","neg","0","false","no"} else None
    
    for hpv_col in ("HPV result", "Dx:HPV", "STDs:HPV"):
        if hpv_col in row_num and hpv_norm is not None:
             row_num[hpv_col] = float(hpv_norm)
        if hpv_col in row_cat and hpv_norm is not None:
            # Check if mapped value is valid for this column
            if cat_choices is not None and hpv_col in cat_cols:
                idx = cat_cols.index(hpv_col)
                # OHE categories are usually strings like '0', '1'
                trained = set(map(str, cat_choices[idx]))

                if hpv_norm in trained:
                     row_cat[hpv_col] = hpv_norm
                elif (hpv_norm + ".0") in trained:
                     row_cat[hpv_col] = hpv_norm + ".0"
                elif hpv_norm == "1" and "True" in trained:
                     row_cat[hpv_col] = "True"
                elif hpv_norm == "0" and "False" in trained:
                     row_cat[hpv_col] = "False"
            else:
                 # Fallback if validation not possible
                 row_cat[hpv_col] = hpv_norm

    # --- Build branch DataFrames in the exact training column order ---
    import pandas as pd, numpy as np
    df_num = pd.DataFrame([row_num], columns=num_cols)
    df_cat = pd.DataFrame([row_cat], columns=cat_cols).astype(str) if cat_cols else pd.DataFrame(index=[0])

    # --- Transform like training ---
    num_imp_arr = num_imp.transform(df_num)
    df_num_imp = pd.DataFrame(num_imp_arr, columns=num_cols, index=df_num.index)

    if cat_cols:
        ohe_out = ohe.transform(df_cat)
        # Handle both sparse and dense outputs
        try:
            ohe_arr = ohe_out.toarray()      # scipy.sparse
        except AttributeError:
            ohe_arr = np.asarray(ohe_out)    # already dense

        # Feature name API across sklearn versions
        if hasattr(ohe, "get_feature_names_out"):
            ohe_cols = list(ohe.get_feature_names_out(cat_cols))
        else:
            ohe_cols = list(ohe.get_feature_names(cat_cols))

        df_cat_ohe = pd.DataFrame(ohe_arr, columns=ohe_cols, index=df_cat.index)
    else:
        df_cat_ohe = pd.DataFrame(index=df_num.index)

    # --- Combine numeric + encoded categoricals to the final layout ---
    X_full = pd.concat([df_num_imp, df_cat_ohe], axis=1)

    # Reindex to exact training order (AFTER OHE) if saved
    if isinstance(feat_after, (list, tuple, np.ndarray)):
        X_full = X_full.reindex(columns=list(feat_after), fill_value=0.0)

    # --- Scale & predict ---
    X_scaled = scaler.transform(X_full)
    prob = float(model.predict_proba(X_scaled)[:, 1][0])

    # --- Age Progressive Penalty ---
    patient_age = float(features.get('age', 0))
    if patient_age > 35:
        years_over = patient_age - 35
        age_penalty = min(0.20, years_over * 0.005) # +0.5% boost per year after 35
        prob = min(1.0, prob + age_penalty)

    label = "High" if prob >= 0.50 else "Low"

    # --- Clinical Rule Override ---
    # HPV Positive is a strong cancer risk factor - ensure High label
    hpv_input = str(features.get("hpv_result", "")).strip().lower()
    is_hpv_positive = hpv_input in {"positive", "pos", "1", "true", "yes"}
    is_hpv_unknown = hpv_input in {"unknown", "", "none", "n/a"}
    
    if is_hpv_positive:
        # HPV+ patients should always be High risk
        prob = 0.5 + (prob * 0.5)
        label = "High" 
    elif is_hpv_unknown:
        pass

    # --- Optional SHAP (non-blocking) ---
    shap_path = ""
    shap_explanation = ""
    try:
        from .shap_explain import generate_shap
        from .llm_explain import generate_explanation
        
        if record_id is not None:
            # pass raw branch frames; SHAP will rebuild the same pipeline internally
            raw_like = pd.concat([df_num, df_cat], axis=1)
            shap_path, top_features = generate_shap(raw_like, record_id=record_id)
            if shap_path: # Only generate explanation if SHAP succeeded
                 shap_explanation = generate_explanation(ui, label)

    except Exception as e:
        print(f"[Warning] SHAP generation failed (non-fatal): {e}")

    return prob, label, shap_path, shap_explanation

# ------------------- Image Prediction ------------------- #
def image_predict(img_path, record_id):
    try:
        model, class_names = load_image_model()
        model.eval()
        img = Image.open(img_path).convert("RGB")
        input_tensor = IMG_TRANSFORM(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            if logits.ndim == 2:
                C = logits.size(1)
            else:
                logits = logits.view(logits.size(0), -1)
                C = logits.size(1)

            if C == 1:
                # binary sigmoid head
                prob_pos = torch.sigmoid(logits)[0, 0].item()
                label = "High" if prob_pos >= 0.50 else "Low"
                display_prob = prob_pos

            elif C == 2:
                # softmax over 2 classes
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                # pick positive index: try to detect in class names; fallback idx=1
                pos_idx = 1
                if class_names and len(class_names) == 2:
                    lower = [str(c).lower() for c in class_names]
                    # heuristics for positive/high/abnormal mapping
                    for i, name in enumerate(lower):
                        if any(k in name for k in ("high", "abnormal", "cancer", "positive", "dys")):
                            pos_idx = i
                            break
                prob_pos = float(probs[pos_idx])
                label = "High" if prob_pos >= 0.50 else "Low"
                display_prob = prob_pos

            else:
                # multi-class: show best class & its probability
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                best_idx = int(probs.argmax())
                display_prob = float(probs[best_idx])
                # if we have names use them, else just High/Low on a 0.5 threshold is meaningless; show class
                label = class_names[best_idx] if class_names and best_idx < len(class_names) else f"class_{best_idx}"

        # Grad-CAM save + path normalisation (keep what you already have)
        filename_base = f"record_{record_id}.png"
        gradcam_save_path_abs = generate_gradcam(img, filename_base)
        # Extract the relative path from 'cervical/static/' onwards
        if gradcam_save_path_abs and 'cervical/static/' in gradcam_save_path_abs.replace('\\', '/'):
            gradcam_static_path = gradcam_save_path_abs.replace('\\', '/').split('cervical/static/')[-1]
        else:
            # Fallback: include gradcam subdirectory
            gradcam_static_path = os.path.join(
                STATIC_PATH_PREFIX, "gradcam", os.path.basename(gradcam_save_path_abs)
            ).replace("\\", "/")

        return float(display_prob), label, gradcam_static_path

    except Exception as e:
        print("❌ Image predict error:", e)
        return 0.0, "Low", ""

# ------------------- Multimodal Prediction ------------------- #
def multimodal_predict(img_path, clinical_features, record_id):
    clin_prob, clin_label, shap_path, shap_explanation = clinical_predict(clinical_features, record_id)
    img_prob, img_label, gradcam_path = image_predict(img_path, record_id)

    fused_score = fuse_probs(clin_prob, img_prob)
    fused_label = "High" if float(fused_score) >= 0.50 else "Low"  # <-- 0.50 threshold

    # Critically override the OVERALL Fused Risk if HPV is Positive
    hpv_input = str(clinical_features.get("hpv_result", "")).strip().lower()
    is_hpv_positive = hpv_input in {"positive", "pos", "1", "true", "yes"}
    if is_hpv_positive:
        fused_score = max(0.99, float(fused_score)) # Ensure mathematically extremely high
        fused_label = "High"

    return {
        "clinical_prob": float(clin_prob),
        "clinical_label": "High" if float(clin_prob) >= 0.50 else "Low",
        "image_prob": float(img_prob),
        "image_label": img_label,
        "fused_score": float(fused_score),
        "fused_label": fused_label,
        "shap_path": shap_path,
        "shap_explanation": shap_explanation,
        "gradcam_path": gradcam_path
    }
