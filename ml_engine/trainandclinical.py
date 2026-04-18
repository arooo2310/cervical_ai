import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Configuration and Paths ---
# Define directories relative to this script's location
# Define directories relative to this script's location
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_SCRIPT_DIR) # .../cervical_multimodal
DATA_PATH = os.path.join(BASE_DIR, "balanced_smote_dataset_all_integer.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True) 

# Helper function to save objects
def save_joblib(obj, filename):
    """Saves an object using joblib."""
    joblib.dump(obj, filename)

# --- Data Loading and Preprocessing ---

def load_clinical_csv(path):
    """Loads the cervical cancer data CSV."""
    df = pd.read_csv(path)
    return df

def prepare_clinical(df, target="Biopsy"):
    """
    Handles imputation, encoding, and scaling of clinical data.
    Saves imputer, encoder, and scaler objects to the models directory.
    """
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    # Removed conversion of categorical columns to strings. Model will train purely on numeric columns natively.

    # Separate numeric and categorical again after conversion
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # 1. Impute numeric features
    num_imp = KNNImputer(n_neighbors=5)
    if num_cols:
        # Fit and transform only on numeric columns
        X[num_cols] = num_imp.fit_transform(X[num_cols])
        save_joblib(num_imp, os.path.join(MODELS_DIR, "clinical_num_imputer.joblib"))

    # 2. Categorical encoding (One-Hot Encoding)
    if cat_cols:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        # Fit and transform on categorical columns
        enc_vals = enc.fit_transform(X[cat_cols])
        
        # Create DataFrame from encoded values with correct column names
        feature_names = enc.get_feature_names_out(cat_cols)
        enc_df = pd.DataFrame(enc_vals, columns=feature_names, index=X.index)
        
        # Combine imputed numeric features and encoded categorical features
        X_processed = pd.concat([X[num_cols], enc_df], axis=1)
        save_joblib(enc, os.path.join(MODELS_DIR, "clinical_ohe.joblib"))
    else:
        X_processed = X[num_cols]

    # 3. Scaling
    scaler = StandardScaler()
    # Fit and transform on the combined, processed feature set
    X_scaled = scaler.fit_transform(X_processed)
    save_joblib(scaler, os.path.join(MODELS_DIR, "clinical_scaler.joblib"))

    # 4. Train/Test Split
    # X_scaled is a NumPy array, so we don't return column names directly
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Store the final feature names for SHAP and prediction (AFTER OHE)
    # This is critical for matching UI input to the model's feature set.
    final_feature_names = X_processed.columns.tolist()
    
    # Save the feature names for reference in UI/SHAP code
    joblib.dump(final_feature_names, os.path.join(MODELS_DIR, "clinical_feature_names.joblib"))

    return X_train, X_test, y_train.values, y_test.values, final_feature_names

# --- Training and Evaluation ---

def train_clinical(csv_path=DATA_PATH, show_feature_importance=True):
    """Loads, prepares, trains the model, and saves all necessary files."""
    print(f"📂 Loading clinical data from: {csv_path}")
    df = load_clinical_csv(csv_path)

    print("⚡ Preparing data (impute, encode, scale, split)...")
    X_train, X_test, y_train, y_test, final_feature_names = prepare_clinical(df)

    print(f"🏋️ Training XGBoost classifier with {len(final_feature_names)} features...")
    model = XGBClassifier(
        eval_metric="logloss",
        n_estimators=200,
        max_depth=4,
        random_state=42,
        use_label_encoder=False
    )
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    acc = accuracy_score(y_test, (preds > 0.005).astype(int))
    print(f"✅ Clinical model AUC: {auc:.4f}, Accuracy: {acc:.4f}")

    model_path = os.path.join(MODELS_DIR, "clinical_xgb.joblib")
    save_joblib(model, model_path)
    print(f"💾 Model saved to: {model_path}")

    # Display and Save Feature Importance
    if show_feature_importance:
        try:
            importances = model.feature_importances_
            feature_series = pd.Series(importances, index=final_feature_names).sort_values(ascending=False).head(20)
            
            plt.figure(figsize=(10, 8))
            feature_series.sort_values(ascending=True).plot(kind='barh')
            plt.title("Top 20 Feature Importance - Clinical XGBoost")
            plt.xlabel("Importance (Gain)")
            plt.ylabel("Features")
            plt.tight_layout()
            
            plot_path = os.path.join(MODELS_DIR, "clinical_feature_importance.png")
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            print(f"📊 Feature importance plot saved to: {plot_path}")
        except Exception as e:
            print(f"⚠️ Could not plot feature importance: {e}")

    return model

if __name__ == "__main__":
    # NOTE: Ensure 'data/cervical-cancer.csv' exists relative to this script.
    train_clinical()