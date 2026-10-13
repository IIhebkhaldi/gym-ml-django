# mlapp/forms.py - CORRECTION COMPLÈTE
from django import forms

class SatisfactionForm(forms.Form):
    # Informations de base
    gender = forms.TypedChoiceField(
        label="Genre",
        choices=[(0, 'Femme'), (1, 'Homme')],
        coerce=int,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    Age = forms.IntegerField(
        label="Âge",
        initial=35,
        min_value=18,
        max_value=80,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '18',
            'max': '80'
        })
    )
    
    abonoment_type = forms.TypedChoiceField(
        label="Type d'abonnement",
        choices=[(0, 'Standard'), (1, 'Premium')],
        coerce=int,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    # Fréquentation
    visit_per_week = forms.IntegerField(
        label="Visites par semaine",
        initial=3,
        min_value=1,
        max_value=7,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '1',
            'max': '7'
        })
    )
    
    days_per_week = forms.IntegerField(
        label="Jours de fréquentation par semaine",
        initial=3,
        min_value=1,
        max_value=7,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '1',
            'max': '7'
        })
    )
    
    # Comportement
    avg_time_in_gym = forms.IntegerField(
        label="Temps moyen en salle (minutes)",
        initial=90,
        min_value=10,
        max_value=180,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '10',
            'max': '180'
        })
    )
    
    avg_time_check_in = forms.IntegerField(
        label="Heure d'arrivée moyenne",
        required=False,
        initial=17,
        min_value=6,
        max_value=23,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '6',
            'max': '23'
        })
    )
    
    avg_time_check_out = forms.IntegerField(
        label="Heure de départ moyenne",
        required=False,
        initial=19,
        min_value=8,
        max_value=24,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '8',
            'max': '24'
        })
    )
    
    # SERVICES - CHANGEMENT IMPORTANT : Utiliser BooleanField au lieu de ChoiceField
    attend_group_lesson = forms.BooleanField(
        label="Participe aux cours collectifs",
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input',
            'role': 'switch'
        })
    )
    
    personal_training = forms.BooleanField(
        label="Coach personnel",
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input',
            'role': 'switch'
        })
    )
    
    drink_abo = forms.BooleanField(
        label="Abonnement boissons",
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input',
            'role': 'switch'
        })
    )
    
    uses_sauna = forms.BooleanField(
        label="Utilise le sauna",
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input',
            'role': 'switch'
        })
    )
    
    # Cours préférés
    fav_group_lesson = forms.CharField(
        label="Cours collectifs préférés",
        required=False,
        initial="",
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Ex: Kickboxen, Yoga, BodyPump...'
        }),
        help_text="Séparez par des virgules. Exemples: Kickboxen, Yoga, BodyPump, Zumba, Running"
    )
    
    def clean(self):
        """Nettoyage personnalisé pour gérer les checkbox"""
        cleaned_data = super().clean()
        
        # S'assurer que les checkbox sont 0 ou 1
        for field in ['attend_group_lesson', 'personal_training', 'drink_abo', 'uses_sauna']:
            if field in cleaned_data:
                cleaned_data[field] = 1 if cleaned_data[field] else 0
            else:
                cleaned_data[field] = 0  # Si non coché, valeur par défaut
        
        # Valeurs par défaut pour les champs optionnels
        if not cleaned_data.get('avg_time_check_in'):
            cleaned_data['avg_time_check_in'] = 17
        
        if not cleaned_data.get('avg_time_check_out'):
            cleaned_data['avg_time_check_out'] = 19
        
        return cleaned_data