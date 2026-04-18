from django.db import models
from django.contrib.auth.models import AbstractUser
from django.conf import settings
from django.utils import timezone

ROLE_CHOICES = (
    ('patient', 'Patient'),
    ('doctor', 'Doctor'),
)

# ---------------- CUSTOM USER ----------------
class User(AbstractUser):
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    email = models.EmailField(unique=True)

    REQUIRED_FIELDS = ['username']
    USERNAME_FIELD = 'email'

    def is_patient(self):
        return self.role == 'patient'

    def is_doctor(self):
        return self.role == 'doctor'

# ---------------- DOCTOR PROFILE ----------------
class DoctorProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='doctor_profile')
    doctor_id = models.CharField(max_length=32, unique=True)
    hospital = models.CharField(max_length=128, blank=True)

    def __str__(self):
        return f"Dr. {self.user.get_full_name() or self.user.email}"

# ---------------- PATIENT PROFILE ----------------
class PatientProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='patient_profile')
    age = models.PositiveIntegerField(null=True, blank=True)
    sex = models.CharField(max_length=10, blank=True)
    blood_group = models.CharField(max_length=5, blank=True)
    assigned_doctor = models.ForeignKey(DoctorProfile, on_delete=models.SET_NULL, null=True, blank=True, related_name='patients')

    def __str__(self):
        return f"{self.user.get_full_name() or self.user.email}"

# ---------------- PATIENT RECORDS ----------------
def pap_image_upload_to(instance, filename):
    return f"pap_images/{instance.patient.user.id}/{filename}"

class PatientRecord(models.Model):
    patient = models.ForeignKey(PatientProfile, on_delete=models.CASCADE, related_name='records')
    created_at = models.DateTimeField(default=timezone.now)
    
    # Age at the time of the test
    age = models.PositiveIntegerField(null=True, blank=True) 
    
    hpv_result = models.CharField(
        max_length=8,
        choices=(('Positive', 'Positive'), ('Negative', 'Negative'), ('Unknown', 'Unknown')),
        default='Unknown'
    )

    smoking_years = models.PositiveIntegerField(
        default=0, 
        verbose_name='Smokes (years)'
    )
    
    contraception_years = models.PositiveIntegerField(
        default=0, 
        verbose_name='Hormonal Contraceptives (years)'
    )
    sexual_partners = models.PositiveIntegerField(
        default=0,
        verbose_name='Number of sexual partners'
    )
    first_sexual_intercourse = models.PositiveIntegerField(
        null=True, blank=True,
        verbose_name='Age of first sexual intercourse'
    )
    num_pregnancies = models.PositiveIntegerField(
        null=True, blank=True,
        verbose_name='Number of pregnancies'
    )
    iud_years = models.PositiveIntegerField(
        default=0,
        verbose_name='IUD (years)'
    )

    clinical_risk_score = models.DecimalField(max_digits=24, decimal_places=18, null=True, blank=True)
    clinical_pred_label = models.CharField(max_length=32, blank=True)
    clinical_shap_path = models.CharField(max_length=256, blank=True)
    shap_explanation = models.TextField(blank=True)

    image = models.ImageField(upload_to=pap_image_upload_to, null=True, blank=True)
    image_prob = models.FloatField(null=True, blank=True)
    image_label = models.CharField(max_length=32, blank=True)
    gradcam_path = models.CharField(max_length=256, blank=True)

    fused_score = models.FloatField(null=True, blank=True)
    fused_label = models.CharField(max_length=32, blank=True)
    
    REFERRAL_STATUS_CHOICES = [
        ('N', 'None'),
        ('R', 'Referred'),
        ('C', 'Consulted'),
    ]
    referral_status = models.CharField(
        max_length=1,
        choices=REFERRAL_STATUS_CHOICES,
        default='N',
        verbose_name='Referral Status'
    )
    referral_date = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.patient.user.email} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

    CLINICAL_THRESHOLD = 0.50

    def _sync_clinical_label(self):
        try:
            score = float(self.clinical_risk_score or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        self.clinical_pred_label = "High" if score >= self.CLINICAL_THRESHOLD else "Low"

    def save(self, *args, **kwargs):
        # If a score is present, always recompute the label from it
        if self.clinical_risk_score is not None:
            self._sync_clinical_label()
        super().save(*args, **kwargs)

# ---------------- PATIENT MESSAGES (Ask a Doubt) ----------------
class PatientDoubt(models.Model):
    record = models.ForeignKey(PatientRecord, on_delete=models.CASCADE, related_name='doubts') 
    sender = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='doubts_sent')
    # 🔑 FIX: The field name used in the view logic must match 'question'
    question = models.TextField() 
    
    answer = models.TextField(blank=True) 
    
    created_at = models.DateTimeField(default=timezone.now)
    
    is_answered = models.BooleanField(default=False) 
    answered_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = "Patient Doubt"
        verbose_name_plural = "Patient Doubts"
        ordering = ['-created_at']

    def __str__(self):
        return f"Doubt by {self.sender.email} on {self.created_at.strftime('%Y-%m-%d')}"
    