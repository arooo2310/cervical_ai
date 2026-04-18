import threading
from decimal import Decimal, InvalidOperation
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages

# UPDATED IMPORTS based on directory rename
from ml_engine.predict_wrappers import multimodal_predict, clinical_predict
from federated.fed_client import train_locally

from ..forms import ClinicalForm, PapImageForm, PatientProfileUpdateForm
from ..models import PatientProfile, PatientRecord, PatientDoubt
from .utils import clean_path

# ---------------- PATIENT VIEWS ----------------

@login_required
def patient_dashboard(request):
    """Displays patient's profile and test history."""
    if request.user.role != 'patient':
        # Fallback in case a doctor ends up here
        return redirect('doctor_dashboard')
    
    profile = get_object_or_404(PatientProfile, user=request.user)
    # Prefetch the doubts for each record for efficient rendering
    records = PatientRecord.objects.filter(patient=profile).prefetch_related('doubts').order_by('-created_at')

    # Calculate weighted average percentage for each record
    for r in records:
        if r.clinical_risk_score is not None and r.fused_score is not None:
            try:
                risk_score = float(r.clinical_risk_score)
                fusion_score = float(r.fused_score)
                
                risk_weight = 0.6
                fusion_weight = 0.4

                weighted_avg = (
                    risk_score * risk_weight +
                    fusion_score * fusion_weight
                ) / (risk_weight + fusion_weight)

                r.final_percentage = weighted_avg * 100
            except (ValueError, TypeError):
                r.final_percentage = None
        else:
            r.final_percentage = None
    
    # Template path will be updated to 'cervical/patient/patient_dashboard.html'
    return render(request, 'cervical/patient/patient_dashboard.html', {
        'profile': profile,
        'records': records
    })

@login_required
def clinical_entry(request):
    """Handles clinical data entry and prediction (without image)."""
    print("Clinical entry accessed by:", request.user.role)
    profile = get_object_or_404(PatientProfile, user=request.user)

    form = ClinicalForm(request.POST or None)

    if request.method == 'POST' and form.is_valid():
        rec = form.save(commit=False)
        rec.patient = profile
        rec.save()

        features = {
            'age': rec.age or 0,
            'hpv_result': rec.hpv_result,
            'smoking': rec.smoking_years,
            'contraception': rec.contraception_years,
            'sexual_history': rec.sexual_partners,
            'first_sexual_intercourse': rec.first_sexual_intercourse or 0,
            'num_pregnancies': rec.num_pregnancies or 0,
            'iud_years': rec.iud_years or 0,
        }
        print(rec.id, "Clinical features:", features)

        # --- Predict (SHAP must be non-fatal) ---
        try:
            score, _label_from_model, shap_path, shap_explanation = clinical_predict(features, record_id=rec.id)
        except Exception as e:
            print(f"[Warning] clinical_predict failed (non-fatal): {e}")
            score, shap_path, shap_explanation = 0.0, "", ""

        try:
            score_dec = Decimal(str(score))        # preserve exact text form
        except InvalidOperation:
            score_dec = Decimal("0")

        rec.clinical_risk_score = score_dec
        rec.clinical_pred_label = "High" if float(score_dec) >= 0.50 else "Low"
        rec.clinical_shap_path = clean_path(shap_path)
        rec.shap_explanation = shap_explanation
        rec.save()

        messages.success(
            request,
            "Clinical analysis complete." + (" SHAP added." if shap_path else " (explainability skipped)")
        )
        return redirect('patient_dashboard')

    return render(request, 'cervical/patient/clinical_form.html', {'form': form}) # Updated path


@login_required
def upload_pap(request):
    """Handles image upload and multimodal prediction."""
    profile = get_object_or_404(PatientProfile, user=request.user)
    form = PapImageForm(request.POST or None, request.FILES or None)

    if request.method == 'POST':
        if form.is_valid():
            rec = form.save(commit=False)
            rec.patient = profile
            rec.save()

            features = {
                'age': rec.age or 0,
                'hpv_result': rec.hpv_result,
                'smoking': rec.smoking_years,
                'contraception': rec.contraception_years,
                'sexual_history': rec.sexual_partners,
                'first_sexual_intercourse': rec.first_sexual_intercourse or 0,
                'num_pregnancies': rec.num_pregnancies or 0,
                'iud_years': rec.iud_years or 0,
            }

            print(f"\n{'='*60}")
            print(f"Starting multimodal prediction for record ID: {rec.id}")
            print(f"Image path: {rec.image.path}")
            print(f"Features: {features}")
            print(f"{'='*60}\n")

            result = multimodal_predict(rec.image.path, features, rec.id)

            print(f"\n{'='*60}")
            print(f"Multimodal prediction results:")
            print(f"  clinical_prob: {result.get('clinical_prob')} (type: {type(result.get('clinical_prob'))})")
            print(f"  image_prob: {result.get('image_prob')} (type: {type(result.get('image_prob'))})")
            print(f"  fused_score: {result.get('fused_score')} (type: {type(result.get('fused_score'))})")
            print(f"  gradcam_path: {result.get('gradcam_path')}")
            print(f"  shap_path: {result.get('shap_path')}")
            print(f"{'='*60}\n")

            # --- scores ---
            rec.clinical_risk_score = float(result.get("clinical_prob", 0.0))
            rec.image_prob  = float(result.get("image_prob", 0.0))
            rec.fused_score         = float(result.get("fused_score", 0.0))

            # --- labels strictly from the scores (0.5) ---
            rec.clinical_pred_label = "High" if rec.clinical_risk_score >= 0.50 else "Low"
            rec.image_label = result.get("image_label") or ("High" if rec.image_prob >= 0.50 else "Low")
            rec.fused_label         = "High" if rec.fused_score         >= 0.50 else "Low"

            # --- paths (relative for {% static %}) ---
            rec.gradcam_path        = clean_path(result.get("gradcam_path") or "")
            rec.clinical_shap_path  = clean_path(result.get("shap_path") or "")
            rec.shap_explanation    = result.get("shap_explanation", "")

            print(f"\n{'='*60}")
            print(f"Saving record with values:")
            print(f"  clinical_risk_score: {rec.clinical_risk_score}")
            print(f"  image_prob: {rec.image_prob}")
            print(f"  fused_score: {rec.fused_score}")
            print(f"  gradcam_path: {rec.gradcam_path}")
            print(f"  clinical_shap_path: {rec.clinical_shap_path}")
            print(f"{'='*60}\n")

            rec.save()

            # --- Federated Learning Trigger ---
            try:
                # Run FL client in a separate thread to avoid blocking the user response
                t = threading.Thread(target=train_locally, args=(rec.patient.id,))
                t.daemon = True
                t.start()
                print(f"Triggered FL training for patient {rec.patient.id}")
            except Exception as e:
                print(f"Failed to trigger FL: {e}")
            # ----------------------------------

            messages.success(request, "Pap image uploaded and analysis complete.")
            return redirect('patient_dashboard')
        else:
            print("FORM IS INVALID. Errors:", form.errors)
            messages.error(request, "Image upload failed. Please check form errors.")

    return render(request, 'cervical/patient/image_upload.html', {'form': form}) # Updated path


@login_required
def patient_detail(request, record_id):
    """
    Displays details for a specific patient record and handles form submissions 
    for asking doubts and referring to a doctor.
    """
    rec = get_object_or_404(PatientRecord, id=record_id)
    
    # Security check: Patient can only view their own records
    if request.user.role == 'patient' and rec.patient.user != request.user:
        messages.error(request, "Access denied.")
        return redirect('patient_dashboard')
        
    if request.method == 'POST' and request.user.role == 'patient':
        action = request.POST.get('action')
        msg = request.POST.get('message', '').strip()

        if action == 'ask_doubt' and msg:
            # 🔑 FIX: Use 'question=msg' to match the PatientDoubt model field
            PatientDoubt.objects.create(
                record=rec, 
                sender=request.user, 
                question=msg  
            )
            messages.success(request, "Your question has been sent to the doctor.")
            
        elif action == 'refer':
            # Note: The model uses referral_status='R', not a boolean 'referred'
            rec.referral_status = 'R'
            rec.save()
            messages.info(request, "This record has been flagged for doctor consultation.")
            
        # Redirect after POST to prevent resubmission
        return redirect('patient_detail', record_id=rec.id)
    
    # Retrieve all doubts related to this record
    doubts = PatientDoubt.objects.filter(record=rec).order_by('-created_at')
            
    # Assuming patient_detail is shared or I should move it. 
    # For now putting it in patient/ though doctor uses it too?
    # Doctor uses 'patient_detail' view too (doctor_view_patient_record actually reuses the template).
    # I should place the template in 'shared' or check if I should split it.
    # The view 'patient_detail' is for patients. 'doctor_view_patient_record' is for doctors.
    # They might use the same template. I will put it in 'shared' if it's reused.
    return render(request, 'cervical/shared/patient_detail.html', {'rec': rec, 'doubts': doubts})


@login_required
def patient_profile_view(request):
    """Display patient's profile information."""
    if request.user.role != 'patient':
        return redirect('doctor_dashboard')
    
    profile = get_object_or_404(PatientProfile, user=request.user)
    
    return render(request, 'cervical/patient/patient_profile.html', {
        'profile': profile,
        'user': request.user
    })


@login_required
def update_patient_profile(request):
    """Handles updating a patient's profile details."""
    profile = get_object_or_404(PatientProfile, user=request.user)
    
    if request.method == 'POST':
        form = PatientProfileUpdateForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, "Profile updated successfully.")
            return redirect('patient_dashboard')
    else:
        form = PatientProfileUpdateForm(instance=profile)
        
    return render(request, 'cervical/patient/patient_update.html', {'form': form}) # Updated path

@login_required
def ask_doubt_view(request):
    """
    Handles POST requests from the 'Ask Doubt' form. 
    It is used only to capture the form data and redirect, 
    as the main logic is handled in patient_detail. 
    """
    if request.user.role != 'patient':
        messages.error(request, "Access denied.")
        return redirect('auth_container')
        
    # Check for POST data submitted by the doubt form
    if request.method == 'POST':
        # The form should include a hidden input for the record ID
        record_id = request.POST.get('record_id') 
        
        # If record_id is present, redirect to the detail view to process the POST request
        # The detail view's logic will handle the actual creation of the PatientDoubt object.
        if record_id:
            # Re-submit the POST data to the patient_detail view
            return redirect('patient_detail', record_id=record_id)
        else:
            messages.error(request, "Missing record ID for doubt submission.")

    # Fallback redirect to the main patient dashboard
    return redirect('patient_dashboard')
