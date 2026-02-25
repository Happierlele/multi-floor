import os
import sys

def find_file(name, path):
    print(f"Searching for {name} in {path}...")
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
    return None

conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)
print(f"Conda prefix: {conda_prefix}")

target = "swrast_dri.so"
found = find_file(target, conda_prefix)

if found:
    print(f"FOUND: {found}")
else:
    print("NOT FOUND")
