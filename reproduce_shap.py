import os
import sys
import django
import traceback

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project_settings.settings')
django.setup()

from ml_engine.predict_wrappers import clinical_predict

# Dummy data
features = {
    "age": 30,
    "hpv_result": "Positive",
    "smoking": 5,
    "contraception": 2,
    "sexual_history": 3
}

print("Calling clinical_predict...")
try:
    # Modify shap_explain to print stack trace if it swallows it?
    # Actually clinical_predict swallows it. 
    # But shap_explain also catches exception and prints "Warning ...".
    # So we should see the warning in stdout.
    prob, label, shap_path, shap_exp = clinical_predict(features, record_id=999)
    print(f"Result: prob={prob}, label={label}, shap_path='{shap_path}'")
    if not shap_path:
        print("SHAP path is empty! Generation failed.")
except Exception as e:
    print(f"Caught top-level exception: {e}")
    traceback.print_exc()
