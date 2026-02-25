
import sys
import os

try:
    print("Attempting to import parameter_clean...")
    import parameter_clean
    print("Successfully imported parameter_clean.")
    print(f"gifs_path is: {parameter_clean.gifs_path}")
except Exception as e:
    print(f"Error importing parameter_clean: {e}")

try:
    print("Attempting to import worker_habitat...")
    import worker_habitat
    print("Successfully imported worker_habitat.")
except Exception as e:
    print(f"Error importing worker_habitat: {e}")
