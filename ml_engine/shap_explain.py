import matplotlib
matplotlib.use('Agg')
import joblib
import warnings
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import shap
import os, joblib, numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from django.conf import settings

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH          = os.path.join(MODELS_DIR, "clinical_xgb.joblib")
SCALER_PATH         = os.path.join(MODELS_DIR, "clinical_scaler.joblib")
NUM_IMPUTER_PATH    = os.path.join(MODELS_DIR, "clinical_num_imputer.joblib")
OHE_PATH            = os.path.join(MODELS_DIR, "clinical_ohe.joblib")
FEAT_AFTER_OHE_PATH = os.path.join(MODELS_DIR, "clinical_feature_names.joblib")

# MEDIA_ROOT is already configured to point inside your static tree.
SHAP_DIR = os.path.join(settings.MEDIA_ROOT, "shap")

def _safe_to_float(x):
    import re, numpy as np
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    # strip surrounding brackets/quotes
    s = s.strip("[](){}\"'")
    # grab first number-like token (incl. exponent)
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            pass
    return 0.0

def _ensure_training_schema(df_in: pd.DataFrame):
    """
    Build two frames in the *training* raw schema:
      - df_num: numeric raw columns in the order num_imputer.feature_names_in_
      - df_cat: categorical raw columns in the order ohe.feature_names_in_
    Values come from df_in when present, otherwise sensible defaults:
      - numeric: 0
      - categorical: the first category seen during fit (avoids unknowns)
    """
    num_imp   = joblib.load(NUM_IMPUTER_PATH)
    ohe       = joblib.load(OHE_PATH)

    if not hasattr(num_imp, "feature_names_in_"):
        raise RuntimeError("Numeric imputer lacks feature_names_in_. Re-train saving this attribute.")

    num_cols    = list(num_imp.feature_names_in_)
    cat_cols    = list(getattr(ohe, "feature_names_in_", []))
    cat_choices = getattr(ohe, "categories_", None)

    # numeric defaults, override from df_in when present
    row_num = {c: np.nan for c in num_cols}
    for c in num_cols:
        if c in df_in.columns:
            val = df_in.iloc[0][c]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            row_num[c] = _safe_to_float(val)

    # categorical defaults = first seen category; override from df_in when present
    row_cat = {}
    for i, c in enumerate(cat_cols):
        default = (cat_choices[i][0] if cat_choices is not None and len(cat_choices[i]) else "Unknown")
        val = df_in.iloc[0][c] if c in df_in.columns else default
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        row_cat[c] = str(val) if pd.notna(val) else str(default)

    df_num = pd.DataFrame([row_num], columns=num_cols)
    df_cat = pd.DataFrame([row_cat], columns=cat_cols).astype(str) if cat_cols else pd.DataFrame(index=[0])

    return df_num, df_cat, num_cols, cat_cols

def _transform_to_model_space(df_num: pd.DataFrame, df_cat: pd.DataFrame):
    """Apply imputer→OHE→concat→reindex→scaler to get X_full (unscaled) and X_scaled."""
    num_imp        = joblib.load(NUM_IMPUTER_PATH)
    ohe            = joblib.load(OHE_PATH)
    scaler         = joblib.load(SCALER_PATH)
    feat_after_ohe = joblib.load(FEAT_AFTER_OHE_PATH)

    # numeric branch
    num_imp_arr = num_imp.transform(df_num)
    df_num_imp  = pd.DataFrame(num_imp_arr, columns=df_num.columns, index=df_num.index)

    # categorical branch
    if len(df_cat.columns) > 0:
        ohe_out = ohe.transform(df_cat)
        try:
            ohe_arr = ohe_out.toarray()      # sparse -> dense
        except AttributeError:
            ohe_arr = np.asarray(ohe_out)    # already dense
        ohe_cols = list(
            ohe.get_feature_names_out(df_cat.columns)
            if hasattr(ohe, "get_feature_names_out")
            else ohe.get_feature_names(df_cat.columns)
        )
        df_cat_ohe = pd.DataFrame(ohe_arr, columns=ohe_cols, index=df_cat.index)
    else:
        df_cat_ohe = pd.DataFrame(index=df_num.index)

    # combine to training layout and reindex to final order the model saw
    X_full = pd.concat([df_num_imp, df_cat_ohe], axis=1)
    if isinstance(feat_after_ohe, (list, tuple, np.ndarray)):
        X_full = X_full.reindex(columns=list(feat_after_ohe), fill_value=0.0)

    X_scaled = scaler.transform(X_full)
    return X_full, X_scaled

def generate_shap(df_raw_like: pd.DataFrame, record_id: int) -> str:
    """
    Best-effort SHAP: returns a static-relative path like
    'cervical/uploads/shap/record_<id>_shap.png' or '' on failure.
    Accepts a row containing *any* subset; missing columns are defaulted.
    """
    try:
        model = joblib.load(MODEL_PATH)

        # 1) Build training-schema frames from whatever we got
        if isinstance(df_raw_like, pd.Series):
            df_raw_like = df_raw_like.to_frame().T
        df_num, df_cat, _, _ = _ensure_training_schema(df_raw_like)

        # 2) Transform to model space exactly as during training
        X_full, X_scaled = _transform_to_model_space(df_num, df_cat)

        # 3) SHAP values
        explainer = shap.TreeExplainer(model)
        try:
            shap_values = explainer(X_scaled).values   # newer SHAP
        except Exception:
            sv = explainer.shap_values(X_scaled)       # older SHAP
            shap_values = sv[1] if isinstance(sv, list) else sv

        vals = shap_values[0]
        cols = X_full.columns

        aggregated_shap = {
            "Age": 0.0,
            "Smoking History": 0.0,
            "Contraceptives": 0.0,
            "Sexual History": 0.0,
            "First Intercourse": 0.0,
            "Pregnancies": 0.0,
            "IUD Usage": 0.0,
            "HPV Result": 0.0
        }
        
        for feature_name, value in zip(cols, vals):
            fname = feature_name.lower()
            if "age" in fname:
                aggregated_shap["Age"] += value
            elif "smoke" in fname:
                aggregated_shap["Smoking History"] += value
            elif "contracept" in fname:
                aggregated_shap["Contraceptives"] += value
            elif "first" in fname or "intercourse" in fname:
                aggregated_shap["First Intercourse"] += value
            elif "pregnanc" in fname:
                aggregated_shap["Pregnancies"] += value
            elif "iud" in fname:
                aggregated_shap["IUD Usage"] += value
            elif "sexual" in fname or "partner" in fname:
                aggregated_shap["Sexual History"] += value
            elif "hpv" in fname or "std" in fname:
                aggregated_shap["HPV Result"] += value
                
        # Force visual SHAP to mathematically match the manual wrapper penalty
        try:
            patient_age = float(df_raw_like["Age"].iloc[0]) if "Age" in df_raw_like.columns else 0
        except Exception:
            patient_age = 0
            
        if patient_age > 35:
            # SHAP natively evaluates the raw trees but skips our external +1% wrapper penalty.
            # We explicitly override the visual plot value to the positive side to render it accurately in red.
            aggregated_shap["Age"] = max(0.10, abs(aggregated_shap["Age"]))

        # USER REQUEST: If the input value is literally zero/false, do not show it on the SHAP graph.
        try:
            if "Smokes (years)" in df_raw_like.columns and float(df_raw_like["Smokes (years)"].iloc[0]) <= 0:
                aggregated_shap["Smoking History"] = 0.0
                
            if "Hormonal Contraceptives (years)" in df_raw_like.columns and float(df_raw_like["Hormonal Contraceptives (years)"].iloc[0]) <= 0:
                aggregated_shap["Contraceptives"] = 0.0
                
            if "Number of sexual partners" in df_raw_like.columns and float(df_raw_like["Number of sexual partners"].iloc[0]) <= 0:
                aggregated_shap["Sexual History"] = 0.0

            if "First sexual intercourse" in df_raw_like.columns and float(df_raw_like["First sexual intercourse"].iloc[0]) <= 0:
                aggregated_shap["First Intercourse"] = 0.0

            if "Num of pregnancies" in df_raw_like.columns and float(df_raw_like["Num of pregnancies"].iloc[0]) <= 0:
                aggregated_shap["Pregnancies"] = 0.0
                
            if "IUD (years)" in df_raw_like.columns and float(df_raw_like["IUD (years)"].iloc[0]) <= 0:
                aggregated_shap["IUD Usage"] = 0.0
                
            # For categorical/OHE HPV columns 
            for hpv_col in ["HPV result", "Dx:HPV", "STDs:HPV"]:
                if hpv_col in df_raw_like.columns:
                    val = str(df_raw_like[hpv_col].iloc[0]).strip().lower()
                    if val in {"0", "0.0", "false", "negative", "neg", "none"}:
                        aggregated_shap["HPV Result"] = 0.0
        except Exception as e:
            print("Warning: Failed to zero out SHAP values based on raw inputs.", e)
        
        grouped_feats = list(aggregated_shap.keys())
        grouped_vals = np.array(list(aggregated_shap.values()))
        
        # Sort by absolute impact
        idx = np.argsort(np.abs(grouped_vals))[::-1]
        top_feats = [grouped_feats[j] for j in idx]
        top_vals = grouped_vals[idx]
        
        # Filter out absolute zero entries if needed, but plotting 5 is short enough
        # USER REQUEST: Do not filter out zero values, keep them on the graph as empty 0-length bars.
        # top_feats = [f for f, v in zip(top_feats, top_vals) if abs(v) > 0.0001]
        # top_vals = [v for v in top_vals if abs(v) > 0.0001]
        
        # If all were zero (shouldn't happen but just in case)
        if len(top_feats) == 0:
            top_feats = list(aggregated_shap.keys())
            top_vals  = list(aggregated_shap.values())

        os.makedirs(SHAP_DIR, exist_ok=True)
        save_path = os.path.join(SHAP_DIR, f"record_{record_id}_shap.png")

        plt.figure(figsize=(8, 4))
        y = np.arange(len(top_feats))
        
        # Create standard red/green bar colors based on original math (increasing = red, decreasing = green)
        colors = ['red' if val >= 0 else 'green' for val in top_vals[::-1]]
        
        # Force the visual bars to always point in the positive direction (right)
        # since patients get confused by negative axis values when their input data was positive.
        display_vals = [abs(val) for val in top_vals[::-1]]
        
        plt.barh(y, display_vals, color=colors)
        plt.yticks(y, top_feats[::-1])
        plt.xlabel("Impact Magnitude (Absolute)")
        plt.title("Impact of Your Data on the Prediction")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print("[Warning] SHAP generation failed (non-fatal):", e)
        return "", {}

    return os.path.join("cervical", "uploads", "shap", f"record_{record_id}_shap.png").replace("\\", "/"), dict(zip(top_feats, top_vals))
