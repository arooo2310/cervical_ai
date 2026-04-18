import os
import joblib

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_joblib(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
