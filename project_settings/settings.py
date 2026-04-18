import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# The BASE_DIR definition using pathlib is preferred
BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'replace-this-in-production'
DEBUG = True
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # 🛑 FIX: Removed redundant 'cervical' entry.
    # We only use the explicit configuration class:
    'cervical.apps.CervicalConfig',
    'rest_framework',
    'crispy_forms',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'project_settings.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [], 
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'project_settings.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Kolkata' 
USE_I18N = True
USE_TZ = True

# --- STATIC & MEDIA CONFIGURATION (For the old path request) ---

STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles_build') 
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'cervical', 'static')]

# MEDIA_ROOT is set to the requested old path (inside static/cervical/uploads)
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'cervical', 'static', 'cervical', 'uploads')

APPEND_SLASH = True

# ---------------- CUSTOM USER ----------------
AUTH_USER_MODEL = 'cervical.User'
LOGIN_URL = 'login' 
LOGIN_REDIRECT_URL = 'patient_dashboard'
LOGOUT_REDIRECT_URL = 'login'

# ---------------- CRISPY FORMS ----------------
CRISPY_ALLOWED_TEMPLATE_PACKS = "bootstrap5"
CRISPY_TEMPLATE_PACK = "bootstrap5"
