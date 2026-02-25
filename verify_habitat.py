import sys
import os

print("----------------------------------------------------------------")
print(f"Python Executable: {sys.executable}")
print("Attempting to import habitat_sim...")

try:
    import habitat_sim
    print(f"✅ SUCCESS: habitat_sim imported. Version: {habitat_sim.__version__}")
except ImportError as e:
    print(f"❌ FAILED: Could not import habitat_sim. Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ FAILED: Unexpected error during import. Error: {e}")
    sys.exit(1)

print("\nChecking build configuration...")
try:
    print(f"  - Built with Bullet Physics: {getattr(habitat_sim, 'built_with_bullet', 'Unknown')}")
    print(f"  - Built with CUDA: {getattr(habitat_sim, 'cuda_enabled', 'Unknown')}")
except Exception as e:
    print(f"⚠️  Warning checking build config: {e}")

print("\nChecking library dependencies (headless rendering)...")
try:
    # Creating a configuration often triggers loading of GL libraries
    # This is the moment where it usually crashes if libEGL/libGL is missing
    cfg = habitat_sim.SimulatorConfiguration()
    print("✅ SUCCESS: SimulatorConfiguration created (GL libraries likely loaded).")
except Exception as e:
    print(f"❌ FAILED: Could not create SimulatorConfiguration. This often indicates missing EGL/GL libraries.")
    print(f"Error: {e}")
    sys.exit(1)

print("\nChecking PyTorch and CUDA...")
try:
    import torch
    print(f"✅ SUCCESS: PyTorch imported. Version: {torch.__version__}")
    print(f"  - CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - Device Count: {torch.cuda.device_count()}")
        print(f"  - Current Device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  WARNING: CUDA is NOT available. Training will be slow on CPU.")
except ImportError:
    print("❌ FAILED: Could not import torch. Please install PyTorch.")
except Exception as e:
    print(f"❌ FAILED: Error checking PyTorch: {e}")

print("\n----------------------------------------------------------------")
print("🎉 CONGRATULATIONS: The Habitat Environment (Sim + PyTorch) appears to be working!")
print("You can now proceed to run your experiments.")
print("----------------------------------------------------------------")
