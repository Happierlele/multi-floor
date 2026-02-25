import sys
import os
print(f"sys.platform: {sys.platform}")
print(f"os.name: {os.name}")
print(f"LD_PRELOAD: {os.environ.get('LD_PRELOAD', 'Not Set')}")
print(f"PATH: {os.environ.get('PATH')}")
try:
    import habitat_sim
    print(f"Habitat-Sim imported: {habitat_sim.__file__}")
except ImportError as e:
    print(f"Habitat-Sim import failed: {e}")
