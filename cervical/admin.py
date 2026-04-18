from django.contrib import admin
from .models import User, DoctorProfile, PatientProfile, PatientRecord, PatientDoubt
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin

@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ('email','role','is_staff','is_superuser')
    search_fields = ('email',)
    ordering = ('email',)

admin.site.register(DoctorProfile)
admin.site.register(PatientProfile)
admin.site.register(PatientRecord)

admin.site.register(PatientDoubt)