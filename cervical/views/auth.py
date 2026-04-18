from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages
from ..forms import PatientSignUpForm, DoctorSignUpForm
from ..models import PatientProfile, DoctorProfile

# ---------------- AUTH VIEWS ----------------

def landing_page(request):
    """Renders the landing/home page."""
    if request.user.is_authenticated:
        if request.user.role == 'patient':
            return redirect('patient_dashboard')
        elif request.user.role == 'doctor':
            return redirect('doctor_dashboard')
    return render(request, 'cervical/pages/home.html')

def about_page(request):
    """Renders the about page."""
    return render(request, 'cervical/pages/about.html')

def auth_container(request):
    """
    Renders the unified login/signup container page. 
    This is the default landing view.
    """
    context = {
        'login_form': AuthenticationForm(),
        'patient_form': PatientSignUpForm(),
        'doctor_form': DoctorSignUpForm(),
        'current_view': 'login'  # Default to showing the Login panel
    }
    # MOVED: auth_container.html should be in cervical/auth/auth_container.html eventually,
    # but based on plan: cervical/templates/cervical/auth/auth_container.html
    # so split path is 'cervical/auth/auth_container.html' if we stick to the plan.
    # WAIT: I haven't moved templates yet (that is next step). 
    # But I should code this assuming the templates WILL be moved.
    # Plan says: "Move root HTML files to auth templates folder" -> these are login.html etc.
    # But auth_container.html is likely an existing template I should check.
    # For now I will point to 'cervical/auth/auth_container.html' assuming I'll move it there.
    return render(request, 'cervical/auth/auth_container.html', context)


def signup_patient(request):
    """Handles Patient signup - redirects to login after successful signup."""
    if request.user.is_authenticated:
        logout(request)

    form = PatientSignUpForm(request.POST or None)
    
    if request.method == 'POST':
        if form.is_valid():
            user = form.save() 
            PatientProfile.objects.get_or_create(user=user)
            messages.success(request, "Signup successful! Please log in.")
            return redirect('auth_container')
        else:
            print("\n--- PATIENT SIGNUP FAILED ---")
            print(form.errors)
            print("-----------------------------\n")
    
    context = {
        'login_form': AuthenticationForm(),
        'patient_form': form,
        'doctor_form': DoctorSignUpForm(),
        'current_view': 'signup_patient'
    }
    return render(request, 'cervical/auth/auth_container.html', context)


def signup_doctor(request):
    """Handles Doctor signup - redirects to login after successful signup."""
    if request.user.is_authenticated:
        logout(request)

    form = DoctorSignUpForm(request.POST or None)
    
    if request.method == 'POST':
        if form.is_valid():
            user = form.save()
            DoctorProfile.objects.get_or_create(user=user)
            messages.success(request, "Signup successful! Please log in.")
            return redirect('auth_container')
        else:
            print("\n--- DOCTOR SIGNUP FAILED ---")
            print(form.errors)
            print("----------------------------\n")
            
    context = {
        'login_form': AuthenticationForm(),
        'patient_form': PatientSignUpForm(),
        'doctor_form': form,
        'current_view': 'signup_doctor'
    }
    return render(request, 'cervical/auth/auth_container.html', context)


def user_login(request):
    """Handles user login."""
    form = AuthenticationForm(request, data=request.POST or None)
    
    if request.method == 'POST':
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            
            if user.role == 'patient':
                return redirect('patient_dashboard') 
            elif user.role == 'doctor':
                return redirect('doctor_dashboard')
            else:
                logout(request)
                return redirect('auth_container')

        else:
            # Login failed - form has errors, they will be displayed in template
            messages.error(request, "Invalid email or password. Please try again.")

    context = {
        'login_form': form,
        'patient_form': PatientSignUpForm(),
        'doctor_form': DoctorSignUpForm(),
        'current_view': 'login'
    }
    return render(request, 'cervical/auth/auth_container.html', context)


def user_logout(request):
    """Logs the user out."""
    logout(request)
    return redirect('home')
