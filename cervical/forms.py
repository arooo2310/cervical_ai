from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import User, PatientProfile, DoctorProfile, PatientRecord

# Define choices here to resolve the AttributeError
HPV_RESULT_CHOICES = (
    ('Positive', 'Positive'),
    ('Negative', 'Negative'),
    ('Unknown', 'Unknown'),
)
SEX_CHOICES = (('Male','Male'),('Female','Female'),('Other','Other'))
BLOOD_GROUPS = (
    ('A+', 'A Positive (A+)'), ('A-', 'A Negative (A-)'),
    ('B+', 'B Positive (B+)'), ('B-', 'B Negative (B-)'),
    ('AB+', 'AB Positive (AB+)'), ('AB-', 'AB Negative (AB-)'),
    ('O+', 'O Positive (O+)'), ('O-', 'O Negative (O-)'),
)


# ---------------- PATIENT SIGNUP ----------------
class PatientSignUpForm(UserCreationForm):
    # FIX: Use a single field to match the template
    full_name = forms.CharField(max_length=128, required=True, label="Legal Full Name")
    
    age = forms.IntegerField(required=True, min_value=0)
    sex = forms.ChoiceField(choices=SEX_CHOICES)
    blood_group = forms.ChoiceField(choices=BLOOD_GROUPS, required=True)
    email = forms.EmailField(required=True, label="Email (Login Username)")

    class Meta:
        model = User
        fields = ('email', 'full_name', 'age', 'sex', 'blood_group')

    def clean_email(self):
        email = self.cleaned_data['email']
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("A user with that email already exists.")
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        
        # FIX: Split the full name into first/last
        full_name = self.cleaned_data.get('full_name', '')
        parts = full_name.split(maxsplit=1)
        user.first_name = parts[0] if parts else ''
        user.last_name = parts[1] if len(parts) > 1 else ''
        
        user.role = 'patient'
        user.username = self.cleaned_data['email'] 
        user.email = self.cleaned_data['email']
        
        if commit:
            user.save()
            PatientProfile.objects.create(
                user=user,
                age=self.cleaned_data['age'],
                sex=self.cleaned_data['sex'],
                blood_group=self.cleaned_data['blood_group']
            )
        return user


# ---------------- DOCTOR SIGNUP ----------------
class DoctorSignUpForm(UserCreationForm):
    # FIX: Use a single field to match the template
    full_name = forms.CharField(max_length=128, required=True, label="Legal Full Name")
    
    email = forms.EmailField(required=True, label="Email (Login Username)")
    doctor_id = forms.CharField(max_length=32, required=True)
    hospital = forms.CharField(max_length=128, required=False)

    class Meta:
        model = User
        fields = ('email', 'full_name', 'doctor_id', 'hospital')
        
    def clean_email(self):
        email = self.cleaned_data['email']
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("A user with that email already exists.")
        return email
        
    def save(self, commit=True):
        user = super().save(commit=False)
        
        # FIX: Split the full name into first/last
        full_name = self.cleaned_data.get('full_name', '')
        parts = full_name.split(maxsplit=1)
        user.first_name = parts[0] if parts else ''
        user.last_name = parts[1] if len(parts) > 1 else ''
        
        user.role = 'doctor'
        user.username = self.cleaned_data['email']
        user.email = self.cleaned_data['email']
        
        if commit:
            user.save()
            DoctorProfile.objects.create(
                user=user,
                doctor_id=self.cleaned_data['doctor_id'],
                hospital=self.cleaned_data['hospital']
            )
        return user


# ---------------- DOCTOR NEW PATIENT FORM (For Quick Creation) ----------------
class DoctorNewPatientForm(forms.Form):
    # User fields
    email = forms.EmailField(required=True, label="Patient Email (Username)")
    first_name = forms.CharField(max_length=64, required=True, label="First Name")
    last_name = forms.CharField(max_length=64, required=True, label="Last Name")

    # PatientProfile fields
    age = forms.IntegerField(required=True, min_value=0, label="Patient Age")
    sex = forms.ChoiceField(choices=SEX_CHOICES, label="Patient Sex")
    blood_group = forms.ChoiceField(choices=BLOOD_GROUPS, required=True, label="Patient Blood Group")


# ---------------- CLINICAL FORM (for PatientRecord) ----------------
class ClinicalForm(forms.ModelForm):
    smoking_years = forms.IntegerField(
        required=False, 
        min_value=0, 
        initial=0, 
        label="Years of Smoking (0 if non-smoker)"
    )
    contraception_years = forms.TypedChoiceField(
        choices=[(0, '0 - No'), (1, '1 - Yes')],
        coerce=int,
        label="Hormonal Contraceptives"
    )
    sexual_partners = forms.IntegerField(
        required=False, 
        min_value=0, 
        initial=0, 
        label="Number of Sexual Partners"
    )
    first_sexual_intercourse = forms.IntegerField(
        required=False,
        min_value=0,
        label="Age of First Sexual Intercourse (0 if never)"
    )
    num_pregnancies = forms.IntegerField(
        required=False,
        min_value=0,
        initial=0,
        label="Number of Pregnancies"
    )
    iud_years = forms.TypedChoiceField(
        choices=[(0, '0 - No'), (1, '1 - Yes')],
        coerce=int,
        label="IUD used"
    )
    
    class Meta:
        model = PatientRecord
        fields = [
            'age', 
            'hpv_result', 
            'smoking_years', 
            'contraception_years', 
            'sexual_partners',
            'first_sexual_intercourse',
            'num_pregnancies',
            'iud_years'
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            if isinstance(field.widget, forms.Select):
                field.widget.attrs.update({'class': 'form-select mb-3'})
            else:
                field.widget.attrs.update({'class': 'form-control mb-3'})

    def clean_age(self):
        age = self.cleaned_data.get('age')
        if age is None or age < 0:
            raise forms.ValidationError("Age must be positive")
        return age


# ---------------- PAP IMAGE UPLOAD (for PatientRecord - combined entry) ----------------
class PapImageForm(forms.ModelForm):
    age = forms.IntegerField(required=True, min_value=0)
    hpv_result = forms.ChoiceField(choices=HPV_RESULT_CHOICES) 
    
    smoking_years = forms.IntegerField(
        required=False, 
        min_value=0, 
        initial=0, 
        label="Years of Smoking (0 if non-smoker)"
    )
    contraception_years = forms.TypedChoiceField(
        choices=[(0, '0 - No'), (1, '1 - Yes')],
        coerce=int,
        label="Hormonal Contraceptives"
    )
    sexual_partners = forms.IntegerField(
        required=False, 
        min_value=0, 
        initial=0, 
        label="Number of Sexual Partners"
    )
    first_sexual_intercourse = forms.IntegerField(
        required=False,
        min_value=0,
        label="Age of First Sexual Intercourse (0 if never)"
    )
    num_pregnancies = forms.IntegerField(
        required=False,
        min_value=0,
        initial=0,
        label="Number of Pregnancies"
    )
    iud_years = forms.TypedChoiceField(
        choices=[(0, '0 - No'), (1, '1 - Yes')],
        coerce=int,
        label="IUD used"
    )

    image = forms.ImageField(required=True, widget=forms.ClearableFileInput(attrs={'accept':'image/*'}))
    
    class Meta:
        model = PatientRecord
        fields = [
            'age', 
            'hpv_result', 
            'smoking_years', 
            'contraception_years', 
            'sexual_partners', 
            'first_sexual_intercourse',
            'num_pregnancies',
            'iud_years',
            'image'
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            if isinstance(field.widget, forms.Select):
                field.widget.attrs.update({'class': 'form-select mb-3'})
            elif isinstance(field.widget, forms.ClearableFileInput):
                field.widget.attrs.update({'class': 'form-control mb-3'})
            else:
                field.widget.attrs.update({'class': 'form-control mb-3'})


# ---------------- PATIENT PROFILE UPDATE ----------------
class PatientProfileUpdateForm(forms.ModelForm):
    first_name = forms.CharField(max_length=30, required=True, label="First Name")
    last_name = forms.CharField(max_length=30, required=True, label="Last Name")

    class Meta:
        model = PatientProfile
        fields = ['first_name', 'last_name', 'age', 'sex', 'blood_group']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.user:
            self.fields['first_name'].initial = self.instance.user.first_name
            self.fields['last_name'].initial = self.instance.user.last_name
            self.fields['sex'].choices = SEX_CHOICES
            self.fields['blood_group'].choices = BLOOD_GROUPS

    def save(self, commit=True):
        profile = super().save(commit=False)
        user = profile.user
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        if commit:
            user.save()
            profile.save()
        return profile


# ---------------- DOCTOR PROFILE UPDATE ----------------
class DoctorProfileUpdateForm(forms.ModelForm):
    first_name = forms.CharField(max_length=30, required=True, label="First Name")
    last_name = forms.CharField(max_length=30, required=True, label="Last Name")

    class Meta:
        model = DoctorProfile
        fields = ['first_name', 'last_name', 'doctor_id', 'hospital']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.user:
            self.fields['first_name'].initial = self.instance.user.first_name
            self.fields['last_name'].initial = self.instance.user.last_name

    def save(self, commit=True):
        profile = super().save(commit=False)
        user = profile.user
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        if commit:
            user.save()
            profile.save()
        return profile