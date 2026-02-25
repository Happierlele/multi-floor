import os
import sys
import glob

print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not Set')}")
print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'Not Set')}")

def find_lib(name, path):
    print(f"Searching for {name} in {path}...")
    # Use glob for faster search in lib dirs
    patterns = [
        os.path.join(path, "lib", name),
        os.path.join(path, "lib", name + "*"),
        os.path.join(path, "lib", "nvidia", name + "*"), # specific to torch/nvidia
        os.path.join(path, "x86_64-conda-linux-gnu", "sysroot", "usr", "lib", name + "*")
    ]
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            return matches[0]
    return None

conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)

egl = find_lib("libEGL.so", conda_prefix)
print(f"libEGL: {egl if egl else 'NOT FOUND'}")

gl = find_lib("libGL.so", conda_prefix)
print(f"libGL: {gl if gl else 'NOT FOUND'}")

try:
    import habitat_sim
    print(f"Habitat-Sim Version: {habitat_sim.__version__}")
    # Try to inspect build config if possible
    if hasattr(habitat_sim, 'built_with_cuda'):
        print(f"Built with CUDA: {habitat_sim.built_with_cuda()}")
except ImportError:
    print("Habitat-Sim not installed")
except Exception as e:
    print(f"Error checking habitat: {e}")
