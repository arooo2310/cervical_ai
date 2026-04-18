from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.db.models import Max, Subquery, OuterRef, Prefetch
from django.utils import timezone

from ml_engine.predict_wrappers import multimodal_predict
from ..forms import DoctorNewPatientForm, PapImageForm
from ..models import PatientProfile, PatientRecord, PatientDoubt, User, DoctorProfile
from .utils import is_doctor, clean_path

# ---------------- DOCTOR VIEWS ----------------

@login_required
@user_passes_test(is_doctor)
def doctor_dashboard(request):
    """Displays doctor dashboard summary and list of patients/messages."""
    # Ensure the doctor check is done via decorator, but keep the redirect fallback
    # if request.user.role != 'doctor':
    #     return redirect('patient_dashboard')
    
    total_patients = PatientProfile.objects.all().count()
    
    # --- FIX 1: Simplified and corrected logic for fetching the LATEST record per patient ---
    latest_records_qs = PatientRecord.objects.filter(patient=OuterRef('id')).order_by('-created_at')
    
    # Filter PatientProfiles that have a PatientRecord whose latest entry is 'High'
    # This uses a subquery to find the fused_label of the latest record
    high_risk_patients = PatientProfile.objects.annotate(
        latest_fused_label=Subquery(latest_records_qs.values('fused_label')[:1])
    ).filter(latest_fused_label='High')

    high_risk_count = high_risk_patients.count()
    
    # Total tests count is better fetched from all records
    total_tests = PatientRecord.objects.count()

    # Messages where is_answered is False
    unanswered_messages = PatientDoubt.objects.filter(is_answered=False).count() 
    
    # Fetch patients and their records (ordered by latest) for the list display
    patients = PatientProfile.objects.all().prefetch_related(
        Prefetch('records', queryset=PatientRecord.objects.order_by('-created_at'))
    ).order_by('user__last_name')
    
    messages_list = PatientDoubt.objects.filter(is_answered=False).select_related('record__patient__user').order_by('-created_at') 

    context = {
        'total_patients': total_patients,
        'high_risk_count': high_risk_count,
        'unanswered_messages': unanswered_messages,
        'total_tests': total_tests, # Added total tests context
        'patients': patients, 
        'messages': messages_list,
    }

    return render(request, 'cervical/doctor/doctor_dashboard.html', context)


@login_required
@user_passes_test(is_doctor)
def doctor_predict(request):
    """Doctor interface to select/create a patient and input data for multimodal prediction."""
    # if request.user.role != 'doctor':
    #     return redirect('patient_dashboard')
    
    all_patients = PatientProfile.objects.all().select_related('user').order_by('user__last_name', 'user__first_name')
    
    new_patient_form = DoctorNewPatientForm(request.POST or None, prefix='new')
    prediction_form = PapImageForm(request.POST or None, request.FILES or None, prefix='predict')

    if request.method == 'POST':
        patient_profile = None

        # --- 1. Identify/Create Patient ---
        # Note: Added check for 'new-email' which indicates the New Patient form was likely submitted
        if 'new-email' in request.POST and new_patient_form.is_valid():
            # ... Patient creation logic remains the same ...
            email = new_patient_form.cleaned_data['email']
            user, created = User.objects.get_or_create(email=email, defaults={
                'username': email, 
                'first_name': new_patient_form.cleaned_data['first_name'],
                'last_name': new_patient_form.cleaned_data['last_name'],
                'role': 'patient',
                'is_active': True
            })
            if created:
                user.set_unusable_password() 
                user.save()
            
            patient_profile, _ = PatientProfile.objects.get_or_create(
                user=user, 
                defaults={
                    'age': new_patient_form.cleaned_data.get('age', 0),
                    'sex': new_patient_form.cleaned_data.get('sex', 'U'),
                    'blood_group': new_patient_form.cleaned_data.get('blood_group', 'U'),
                }
            )
            messages.success(request, f"Patient {user.email} selected/created successfully.")
            
        elif request.POST.get('patient_select'):
            patient_id = request.POST.get('patient_select')
            patient_profile = get_object_or_404(PatientProfile, id=patient_id)
        
        # --- 2. Run Prediction ---
        if patient_profile:
            # Re-validate the prediction form after determining the patient
            if prediction_form.is_valid():
                rec = prediction_form.save(commit=False)
                rec.patient = patient_profile
                rec.save() 
                
                img_path = rec.image.path
                features = {
                    'age': rec.age or 0,
                    'hpv_result': rec.hpv_result,
                    'smoking': rec.smoking_years,
                    'contraception': rec.contraception_years,
                    'sexual_history': rec.sexual_partners,
                }
                
                # Assuming multimodal_predict is available
                result = multimodal_predict(img_path, features, rec.id)

                rec.clinical_risk_score = result.get("clinical_prob")
                rec.clinical_pred_label = result.get("clinical_label")
                rec.image_prob = result.get("image_prob")
                rec.image_label = result.get("image_label")
                rec.fused_score = result.get("fused_score")
                rec.fused_label = result.get("fused_label")
                
                # --- FIX 2: Apply clean_path here! ---
                rec.gradcam_path = clean_path(result.get("gradcam_path"))
                rec.clinical_shap_path = clean_path(result.get("shap_path")) 

                rec.save()
                messages.success(request, f"Prediction complete for {patient_profile.user.email}.")
                return redirect('doctor_view_patient_record', record_id=rec.id)
            else:
                # If prediction form fails, re-render with errors
                messages.error(request, "Prediction form validation failed. Please correct the fields.")
                
        else:
            messages.error(request, "Patient was not selected or created correctly. Please check all forms.")

    context = {
        'patients': all_patients,
        'prediction_form': prediction_form,
        'new_patient_form': new_patient_form,
    }
    return render(request, 'cervical/doctor/doctor_predict.html', context)


@login_required
@user_passes_test(is_doctor)
def doctor_view_patient_record(request, record_id):
    """Doctor view of patient record and message reply handling."""
    # if request.user.role != 'doctor':
    #     return redirect('patient_dashboard')
        
    rec = get_object_or_404(PatientRecord.objects.select_related('patient__user'), id=record_id)
    
    if request.method == 'POST':
        if 'reply_message' in request.POST:
            msg_id = request.POST.get('message_id')
            reply_text = request.POST.get('reply_text', '').strip()
            
            if reply_text:
                try:
                    msg = PatientDoubt.objects.get(id=msg_id) 
                    msg.answer = reply_text
                    msg.is_answered = True
                    msg.answered_at = timezone.now()
                    msg.save()
                    messages.success(request, "Response sent successfully.")
                except PatientDoubt.DoesNotExist:
                    messages.error(request, "Message not found.")
            else:
                messages.warning(request, "Reply text cannot be empty.")
            
    # Retrieve all doubts related to this record
    doubts = PatientDoubt.objects.filter(record=rec).order_by('-created_at')
    
    # Reusing the shared template
    return render(request, 'cervical/shared/patient_detail.html', {'rec': rec, 'doubts': doubts})


@login_required
@user_passes_test(is_doctor)
def doctor_messages_view(request):
    """Dedicated page for doctors to view and manage all patient doubts/messages."""
    # Filter parameter for answered/unanswered
    filter_type = request.GET.get('filter', 'all')
    
    # Base queryset with related data for efficiency
    doubts_qs = PatientDoubt.objects.select_related(
        'record__patient__user', 
        'sender'
    ).order_by('-created_at')
    
    # Apply filter
    if filter_type == 'unanswered':
        doubts_qs = doubts_qs.filter(is_answered=False)
    elif filter_type == 'answered':
        doubts_qs = doubts_qs.filter(is_answered=True)
    # 'all' shows everything
    
    # Get counts for filter tabs
    total_count = PatientDoubt.objects.count()
    unanswered_count = PatientDoubt.objects.filter(is_answered=False).count()
    answered_count = PatientDoubt.objects.filter(is_answered=True).count()
    
    context = {
        'doubts': doubts_qs,
        'filter_type': filter_type,
        'total_count': total_count,
        'unanswered_count': unanswered_count,
        'answered_count': answered_count,
    }
    
    return render(request, 'cervical/doctor/doctor_messages.html', context)


@login_required
@user_passes_test(is_doctor)
def doctor_reply_doubt(request, doubt_id):
    """Handle doctor replies to patient doubts."""
    if request.method != 'POST':
        messages.error(request, "Invalid request method.")
        return redirect('doctor_messages')
    
    doubt = get_object_or_404(PatientDoubt, id=doubt_id)
    reply_text = request.POST.get('reply_text', '').strip()
    
    if not reply_text:
        messages.warning(request, "Reply text cannot be empty.")
        return redirect('doctor_messages')
    
    # Save the reply
    doubt.answer = reply_text
    doubt.is_answered = True
    doubt.answered_at = timezone.now()
    doubt.save()
    
    messages.success(request, f"Reply sent successfully to {doubt.sender.email}.")
    
    # Redirect back to messages page, preserving filter if any
    filter_param = request.POST.get('filter', 'all')
    return redirect(f"{request.build_absolute_uri('/doctor/messages/')}?filter={filter_param}".replace(request.build_absolute_uri('/'), '/'))


@login_required
@user_passes_test(is_doctor)
def doctor_profile_view(request):
    """Display doctor's profile information."""
    doctor_profile = get_object_or_404(DoctorProfile, user=request.user)
    
    return render(request, 'cervical/doctor/doctor_profile.html', {
        'doctor_profile': doctor_profile,
        'user': request.user
    })


@login_required
@user_passes_test(is_doctor)
def doctor_profile_update(request):
    """Handle doctor profile updates."""
    from ..forms import DoctorProfileUpdateForm
    
    doctor_profile = get_object_or_404(DoctorProfile, user=request.user)
    
    if request.method == 'POST':
        form = DoctorProfileUpdateForm(request.POST, instance=doctor_profile)
        if form.is_valid():
            form.save()
            messages.success(request, "Profile updated successfully.")
            return redirect('doctor_profile')
    else:
        form = DoctorProfileUpdateForm(instance=doctor_profile)
    
    return render(request, 'cervical/doctor/doctor_profile_update.html', {'form': form})
