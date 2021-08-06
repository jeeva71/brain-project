from django import forms
from .models import *

class brainForm(forms.ModelForm):

    class Meta():
        model = brainModel
        fields = ['upload_file']
