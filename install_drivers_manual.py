import os
import sys
import json
import requests
import tarfile
import shutil
import zipfile
import subprocess
from io import BytesIO

# Try to import zstandard, install if missing
try:
    import zstandard as zstd
except ImportError:
    print("[INFO] 'zstandard' library not found. Attempting to install via pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "zstandard"])
        import zstandard as zstd
        print("[SUCCESS] Installed 'zstandard'")
    except Exception as e:
        print(f"[WARNING] Could not install 'zstandard': {e}")
        print("Processing of .conda files will fail.")
        zstd = None

def install_package_manually(package_name, channel="conda-forge", platform="linux-64"):
    print(f"[INFO] Fetching metadata for {package_name} from {channel}...")
    try:
        r = requests.get(f"https://api.anaconda.org/package/{channel}/{package_name}", timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch metadata for {package_name}: {e}")
        return False

    files = data.get('files', [])
    candidates = []
    
    # Filter for platform and extension
    for f in files:
        attrs = f.get('attrs', {})
        basename = f.get('basename', '')
        
        # Check platform
        if attrs.get('subdir') != platform:
            continue
            
        # Extension check: support .tar.bz2 and .conda (if zstd available)
        if basename.endswith('.tar.bz2'):
            candidates.append(f)
        elif basename.endswith('.conda') and zstd is not None:
            candidates.append(f)
            
    if not candidates:
        print(f"[ERROR] No {platform} packages found for {package_name} (checked .tar.bz2 and .conda) in {channel}")
        return False

    # Sort by version (descending) and build number
    def parse_version(v_str):
        try:
            # Handle versions like "1.2.3a" or "2021.1" by extracting numbers
            parts = []
            current_num = ""
            for char in v_str:
                if char.isdigit():
                    current_num += char
                else:
                    if current_num:
                        parts.append(int(current_num))
                        current_num = ""
            if current_num:
                parts.append(int(current_num))
            return tuple(parts)
        except:
            return (0,)

    candidates.sort(key=lambda x: (parse_version(x.get('version', '0')), x.get('attrs', {}).get('build_number', 0)), reverse=True)
    
    # Pick the latest
    best_candidate = candidates[0]
    download_url = best_candidate.get('download_url')
    if download_url.startswith('//'):
        download_url = 'https:' + download_url
        
    print(f"[INFO] Selected {best_candidate.get('basename')} (Version: {best_candidate.get('version')})")
    print(f"[INFO] Downloading from {download_url}...")
    
    try:
        # Download to a temporary file first (essential for .conda and seeking)
        import tempfile
        suffix = ".conda" if download_url.endswith(".conda") else ".tar.bz2"
        
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_dl:
            print(f"[INFO] Downloading to {tmp_dl.name}...")
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                shutil.copyfileobj(r.raw, tmp_dl)
            tmp_dl_path = tmp_dl.name

        conda_prefix = os.environ.get("CONDA_PREFIX")
        if not conda_prefix:
            print("[ERROR] CONDA_PREFIX not set. Please activate your environment.")
            os.remove(tmp_dl_path)
            return False

        print(f"[INFO] Extracting to {conda_prefix}...")
        
        filename = best_candidate.get('basename')
        
        try:
            if filename.endswith('.tar.bz2'):
                with tarfile.open(tmp_dl_path, mode="r:bz2") as tar:
                    members = [m for m in tar.getmembers() if not m.name.startswith("info/")]
                    tar.extractall(path=conda_prefix, members=members)
                
            elif filename.endswith('.conda'):
                # .conda format: ZIP containing 'pkg-*.tar.zst'
                with zipfile.ZipFile(tmp_dl_path) as zf:
                    for zname in zf.namelist():
                        if zname.startswith("pkg-") and zname.endswith(".tar.zst"):
                            # Extract the zst file to a temp tar file
                            with zf.open(zname) as zst_file:
                                dctx = zstd.ZstdDecompressor()
                                with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp_tar:
                                    with dctx.stream_reader(zst_file) as reader:
                                        shutil.copyfileobj(reader, tmp_tar)
                                    tmp_tar_path = tmp_tar.name
                                    
                                try:
                                    with tarfile.open(tmp_tar_path, mode="r") as tar:
                                         members = [m for m in tar.getmembers() if not m.name.startswith("info/")]
                                         tar.extractall(path=conda_prefix, members=members)
                                finally:
                                    if os.path.exists(tmp_tar_path):
                                        os.remove(tmp_tar_path)
        finally:
             if os.path.exists(tmp_dl_path):
                os.remove(tmp_dl_path)
            
        print(f"[SUCCESS] Installed {package_name}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download/extract: {e}")
        # Print full traceback for debugging
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("==================================================")
    print("   Manual Driver Installer (Bypassing Conda Solver)")
    print("==================================================")
    
    if not os.environ.get("CONDA_PREFIX"):
        print("ERROR: Please activate 'habitat_headless' environment first!")
        sys.exit(1)

    # List of packages to install
    # Use 'conda-forge' for mesa now that we support .conda, but keep anaconda fallback just in case
    packages = [
        # Try conda-forge mesa first (it's newer)
        ("mesa", "conda-forge"),
        
        # If that failed (it did for user), try anaconda channel mesa (older but reliable .tar.bz2)
        # Note: In the loop we handle failure gracefully, so adding both is fine if the first fails.
        # But wait, install_package_manually returns False on failure.
        
        ("libglvnd", "conda-forge"),      # GL Vendor Neutral Dispatch
    ]

    # Special handling for mesa: try conda-forge, if fail, try anaconda
    mesa_success = False
    print("Installing mesa/drivers...")
    if install_package_manually("mesa", channel="conda-forge"):
        mesa_success = True
    elif install_package_manually("mesa-dri-drivers", channel="conda-forge"):
        print("[INFO] Fallback: Installed mesa-dri-drivers from conda-forge")
        mesa_success = True
    elif install_package_manually("mesalib", channel="anaconda"):
        print("[INFO] Fallback: Installed mesalib from anaconda")
        mesa_success = True
        
    if not mesa_success:
        print("WARNING: Failed to find mesa/drivers on conda-forge or anaconda.")

    # Install libllvm (Critical dependency for swrast)
    print("\n--------------------------------------------------")
    print("Installing libllvm (Critical for software rendering)...")
    if not install_package_manually("libllvm", channel="conda-forge"):
        print("WARNING: Failed to find libllvm on conda-forge. Trying anaconda...")
        if not install_package_manually("libllvm", channel="anaconda"):
             print("WARNING: Failed to find libllvm on anaconda. Trying 'llvm' package...")
             install_package_manually("llvm", channel="anaconda")
        
    # Install other packages
    for pkg_name, channel in packages[1:]:
        print(f"\n--------------------------------------------------")
        print(f"Installing {pkg_name} from {channel}...")
        install_package_manually(pkg_name, channel=channel)
        
    print("\n[VERIFICATION]")
    # Search for any swrast_dri.so in the conda prefix to find where it landed
    print("Searching for swrast_dri.so in conda environment...")
    found_drivers = []
    for root, dirs, files in os.walk(os.environ["CONDA_PREFIX"]):
        if "swrast_dri.so" in files:
            full_path = os.path.join(root, "swrast_dri.so")
            found_drivers.append(full_path)
            print(f" - Found at: {full_path}")
            
    if found_drivers:
        print(f"Success! Driver(s) found.")
        print("You can now run 'qsub run.pbs'")
    else:
        print("Error: Driver file still missing after installation.")
        # Try listing contents of lib directory to see structure
        lib_dir = os.path.join(os.environ["CONDA_PREFIX"], "lib")
        if os.path.exists(lib_dir):
            print(f"Contents of {lib_dir} (first 20):")
            try:
                print(os.listdir(lib_dir)[:20])
            except:
                pass
