# src/train_clinical_pipeline.py
import os, joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


BASE_DIR   = os.path.dirname(os.path.dirname(__file__)) 
DATA_PATH  = os.path.join(BASE_DIR, "balanced_smote_dataset_all_integer.csv")  
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# EDIT THIS to your real target column (e.g., "Biopsy" in the UCI dataset)
target_col = "Biopsy"

# Map CSV headers → the UI field names you collect
rename = {
    "Age": "age",
    "Smokes (years)": "smoking",
    "Hormonal Contraceptives (years)": "contraception",
    "Number of sexual partners": "sexual_history",
    "Dx:HPV": "hpv_result",
}

def main():
    df = pd.read_csv(DATA_PATH)

    # keep target, rename to UI keys for the features
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {DATA_PATH}")

    df = df.rename(columns=rename)

    # we’ll only train on the inputs your form actually sends
    candidate = ["age", "hpv_result", "smoking", "contraception", "sexual_history"]
    features = [c for c in candidate if c in df.columns]  # allow hpv_result to be absent
    if not features:
        raise ValueError("None of the expected UI fields were found after renaming.")

    X = df[features].copy()
    y = df[target_col].astype(int)

    # identify types
    numeric = [c for c in ["age","smoking","contraception","sexual_history"] if c in X.columns]
    categorical = [c for c in ["hpv_result"] if c in X.columns]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ],
        remainder="drop",
    )

    clf = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        eval_metric="logloss",
    )

    pipe = Pipeline(steps=[("prep", preprocess), ("clf", clf)])
    pipe.fit(X, y)

    joblib.dump(pipe, os.path.join(MODELS_DIR, "clinical_pipeline.joblib"))
    print("✅ Saved:", os.path.join(MODELS_DIR, "clinical_pipeline.joblib"))

if __name__ == "__main__":
    main()
