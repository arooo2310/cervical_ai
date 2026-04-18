# cervical/urls.py (UPDATED)

from django.urls import path
from .import views

urlpatterns = [
    # Auth & home
    path('', views.landing_page, name='home'),
    path('auth/', views.auth_container, name='auth_container'),

    path('home/', views.landing_page, name='home'),
    path('about/', views.about_page, name='about'),
    path('signup/patient/', views.signup_patient, name='signup_patient'),
    path('signup/doctor/', views.signup_doctor, name='signup_doctor'),
    
    # FIX: Rename the URL name to 'login' for consistency
    path('login/', views.user_login, name='login'), 
    
    path('logout/', views.user_logout, name='logout'),

    # Doctor Messages
    path('doctor/messages/', views.doctor_messages_view, name='doctor_messages'),
    path('doctor/messages/reply/<int:doubt_id>/', views.doctor_reply_doubt, name='doctor_reply_doubt'),

    # Existing record view path (needed for the reply form action)
    path('doctor/record/<int:record_id>/', views.doctor_view_patient_record, name='doctor_view_patient_record'),

    # Patient
    path('patient/dashboard/', views.patient_dashboard, name='patient_dashboard'),
    path('patient/profile/', views.patient_profile_view, name='patient_profile'),
    path('patient/profile/edit/', views.update_patient_profile, name='update_patient_profile'),
    path('patient/clinical/', views.clinical_entry, name='clinical_entry'),
    path('patient/upload/', views.upload_pap, name='upload_pap'),
    path('patient/record/<int:record_id>/', views.patient_detail, name='patient_detail'),

    # Doctor
    path('doctor/dashboard/', views.doctor_dashboard, name='doctor_dashboard'),
    path('doctor/profile/', views.doctor_profile_view, name='doctor_profile'),
    path('doctor/profile/edit/', views.doctor_profile_update, name='doctor_profile_update'),
    path('doctor/predict/', views.doctor_predict, name='doctor_predict'),
    path('doctor/record/<int:record_id>/', views.doctor_view_patient_record, name='doctor_view_patient_record'),

    path('patient/doubt/', views.ask_doubt_view, name='ask_doubt'), 
]