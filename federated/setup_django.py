
import os
import django
import sys
from pathlib import Path

def setup_django_environment():
    """
    Sets up the Django environment for standalone scripts.
    """
    # 1. Add project root to sys.path
    # 'cervical_multimodal/fedrated/setup_django.py' -> 'cervical_multimodal' -> 'cervical-ai' (root)
    # The structure seems to be:
    # .../cervical-ai (base)
    #    /cervical_multimodal (app container? or source?)
    #       /cervical
    #       /fedrated
    #       manage.py (likely in cervical_multimodal or one level up)
    
    # Based on file listing:
    # c:/Users/sbine/Desktop/projects/cervical-ai-final/cervical-ai/cervical-ai/cervical_multimodal/fedrated/
    
    current_file = Path(__file__).resolve()
    fedrated_dir = current_file.parent
    cervical_multimodal_dir = fedrated_dir.parent
    # Assuming manage.py is in cervical_multimodal, or settings is accessible.
    
    # Append the directory containing 'cervical' package
    sys.path.append(str(cervical_multimodal_dir))
    
    # 2. Set environment variable
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project_settings.settings')
    
    # 3. Setup Django
    django.setup()

if __name__ == "__main__":
    setup_django_environment()
    print("Django environment set up successfully!")
