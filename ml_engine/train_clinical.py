import os
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from src.clinical_data_prep import load_clinical_csv, prepare_clinical
from src.utils import ensure_dir, save_joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "balanced_smote_dataset_all_integer.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ensure_dir(MODELS_DIR)
def train_clinical(csv_path=DATA_PATH, show_feature_importance=True):
    print(f"📂 Loading clinical data from: {csv_path}")
    df = load_clinical_csv(csv_path)

    print("⚡ Preparing data (impute, encode, scale, split)...")
    X_train, X_test, y_train, y_test = prepare_clinical(df)

    print("🏋️ Training XGBoost classifier...")
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
    acc = accuracy_score(y_test, (preds > 0.5).astype(int))
    print(f"✅ Clinical model AUC: {auc:.4f}, Accuracy: {acc:.4f}")

    model_path = os.path.join(MODELS_DIR, "clinical_xgb.joblib")
    save_joblib(model, model_path)
    print(f"💾 Model saved to: {model_path}")

    if show_feature_importance:
        try:
            importances = model.feature_importances_
            features = df.drop(columns=["Biopsy"]).columns
            plt.figure(figsize=(10,6))
            plt.barh(features, importances)
            plt.title("Feature Importance - Clinical XGBoost")
            plt.xlabel("Importance")
            plt.ylabel("Features")
            plt.tight_layout()
            plot_path = os.path.join(MODELS_DIR, "clinical_feature_importance.png")
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"⚠️ Could not plot feature importance: {e}")

    return model