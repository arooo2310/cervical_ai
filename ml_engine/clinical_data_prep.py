import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_joblib, ensure_dir
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
ensure_dir(MODELS_DIR)

def load_clinical_csv(path):
    df = pd.read_csv(path)
    return df

def prepare_clinical(df, target="Biopsy"):
    # identify numeric and categorical
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

    # Impute numeric
    num_imp = SimpleImputer(strategy="median")
    if len(num_cols):
        X[num_cols] = num_imp.fit_transform(X[num_cols])
    save_joblib(num_imp, os.path.join(MODELS_DIR, "clinical_num_imputer.joblib"))

    # Categorical encoding (if present)
    if len(cat_cols):
        X[cat_cols] = X[cat_cols].fillna("unknown")
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        enc_vals = enc.fit_transform(X[cat_cols])
        enc_df = pd.DataFrame(enc_vals, columns=enc.get_feature_names_out(cat_cols), index=X.index)
        X = pd.concat([X[num_cols].reset_index(drop=True), enc_df.reset_index(drop=True)], axis=1)
        save_joblib(enc, os.path.join(MODELS_DIR, "clinical_ohe.joblib"))
    else:
        # ensure X is DataFrame numeric
        X = X[num_cols]

    # scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    save_joblib(scaler, os.path.join(MODELS_DIR, "clinical_scaler.joblib"))

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    return X_train, X_test, y_train.values, y_test.values
