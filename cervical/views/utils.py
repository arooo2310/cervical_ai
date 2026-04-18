from pathlib import Path
from django.conf import settings

# ---------------- UTILS ----------------

def clean_path(full_path):
    """
    Converts a full file system path to a URL-relative path expected by {% static %}.
    Finds the first instance of 'cervical/static/' or 'static/' and strips everything before it.
    """
    if not full_path:
        return ""
    
    # Normalize and convert to Path object for reliable splitting
    p = Path(full_path).as_posix()
    
    # --- DEBUGGING STEP ---
    # print(f"\n--- PATH DEBUG ---")
    # print(f"Input Path: {p}")
    # ----------------------
    
    static_prefix = 'cervical/static/'
    if static_prefix in p:
        relative_path = p.split(static_prefix, 1)[-1]
    
    # 2. If that fails, try finding 'static/' (7 chars)
    elif 'static/' in p:
        relative_path = p.split('static/', 1)[-1]
        
    else:
        # 3. If no recognizable prefix is found, save the input path
        relative_path = p
        
    # --- DEBUGGING STEP ---
    # print(f"Saved Path: {relative_path}")
    # print(f"-------------------\n")
    # ----------------------

    return relative_path

def is_doctor(user):
    """Helper function to restrict access to users with the 'doctor' role"""
    return user.is_authenticated and user.role == 'doctor'
