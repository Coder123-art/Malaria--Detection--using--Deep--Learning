#!/usr/bin/env python
import sys
print("Starting test...")
sys.stdout.flush()

try:
    print("Importing app...")
    sys.stdout.flush()
    import app
    print("App imported successfully!")
    sys.stdout.flush()
    
    # Test dataset stats endpoint directly
    with app.app.app_context():
        print("\nTesting dataset stats...")
        sys.stdout.flush()
        # Line 17 in test_app.py
        result = app.dataset_stats()
        print(f"Result: {result}")
        sys.stdout.flush()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()

print("\nDone!")
sys.stdout.flush()
