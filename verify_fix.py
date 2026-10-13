
import os
import sys
import django

# Setup Django environment
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProjetMLDjango.settings')
django.setup()

from mlapp.views import creer_modele_satisfaction_factice, creer_modele_satisfaction

print("Testing Fake Model Creation...")
try:
    model, scaler, features = creer_modele_satisfaction_factice()
    if model:
        print("✅ Fake Model created successfully!")
    else:
        print("❌ Fake Model creation returned None")
except Exception as e:
    print(f"❌ Fake Model creation failed with error: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting Real Model Creation (attempt)...")
try:
    # This might fail if CSV is missing, but we want to see if it logic-errors
    model, scaler, features = creer_modele_satisfaction()
    if model:
        print("✅ Real Model created successfully!")
    else:
        print("⚠️ Real Model creation returned None (might be due to missing CSV, which is expected if no data)")
except Exception as e:
    print(f"❌ Real Model creation failed with error: {e}")
    import traceback
    traceback.print_exc()
