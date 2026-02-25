import os
import ctypes
from ctypes import *
import sys

# Setup Environment (Mirroring our best guess)
os.environ["CUDA_VISIBLE_DEVICES"] = "" # Unset to allow EGL to see GPUs
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

# Preload (Mirroring test_habitat_headless.py)
libs = [
    # "/usr/lib64/libEGL_nvidia.so.0", # Commented out to test natural loading
    # "/usr/lib64/libOpenGL.so.0",
    "/usr/lib64/libEGL.so.1",
    "/usr/lib64/libEGL.so",
]
glcore = "/usr/lib64/libnvidia-glcore.so.575.57.08"
# if os.path.exists(glcore):
#     libs.insert(0, glcore)

for lib in libs:
    if os.path.exists(lib):
        try:
            CDLL(lib, mode=RTLD_GLOBAL)
            print(f"Loaded {lib}")
        except Exception as e:
            print(f"Failed to load {lib}: {e}")

# Load EGL
try:
    egl = CDLL("libEGL.so.1", mode=RTLD_GLOBAL)
except:
    print("Failed to load libEGL.so.1")
    sys.exit(1)

# EGL Constants
EGL_DEFAULT_DISPLAY = 0
EGL_NO_DISPLAY = 0
EGL_NO_CONTEXT = 0
EGL_NO_SURFACE = 0
EGL_TRUE = 1
EGL_FALSE = 0
EGL_SUCCESS = 0x3000

EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_PLATFORM_SURFACELESS_MESA = 0x31DD # Not standard EGL, but common

EGL_BLUE_SIZE = 0x3022
EGL_GREEN_SIZE = 0x3023
EGL_RED_SIZE = 0x3024
EGL_DEPTH_SIZE = 0x3025
EGL_SURFACE_TYPE = 0x3033
EGL_NONE = 0x3038
EGL_RENDERABLE_TYPE = 0x3040
EGL_OPENGL_ES3_BIT = 0x00000040
EGL_OPENGL_BIT = 0x0008
EGL_PBUFFER_BIT = 0x0001

EGL_CONTEXT_CLIENT_VERSION = 0x3098
EGL_CONTEXT_MAJOR_VERSION = 0x3098
EGL_CONTEXT_MINOR_VERSION = 0x30FB
EGL_CONTEXT_OPENGL_PROFILE_MASK = 0x30FD
EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT = 0x00000001

EGL_OPENGL_API = 0x30A2
EGL_OPENGL_ES_API = 0x30A0

# Define argument types
egl.eglGetDisplay.argtypes = [c_void_p]
egl.eglGetDisplay.restype = c_void_p

egl.eglInitialize.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
egl.eglInitialize.restype = c_int

egl.eglChooseConfig.argtypes = [c_void_p, POINTER(c_int), POINTER(c_void_p), c_int, POINTER(c_int)]
egl.eglChooseConfig.restype = c_int

egl.eglBindAPI.argtypes = [c_uint]
egl.eglBindAPI.restype = c_int

egl.eglCreateContext.argtypes = [c_void_p, c_void_p, c_void_p, POINTER(c_int)]
egl.eglCreateContext.restype = c_void_p

egl.eglMakeCurrent.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p]
egl.eglMakeCurrent.restype = c_int

egl.eglGetError.restype = c_int

egl.eglQueryString.argtypes = [c_void_p, c_int]
egl.eglQueryString.restype = c_char_p

# 1. Get Display
print("Calling eglGetDisplay(EGL_DEFAULT_DISPLAY)...")
dpy = egl.eglGetDisplay(EGL_DEFAULT_DISPLAY)
if not dpy:
    print(f"eglGetDisplay failed: {hex(egl.eglGetError())}")
    # Try getting device
    # (Skip complex device enumeration for now, assume default display works or fails)
else:
    print(f"Got Display: {dpy}")

# 2. Initialize
major = c_int()
minor = c_int()
print("Calling eglInitialize...")
if not egl.eglInitialize(dpy, byref(major), byref(minor)):
    print(f"eglInitialize failed: {hex(egl.eglGetError())}")
    sys.exit(1)
print(f"EGL Initialized: {major.value}.{minor.value}")

# 3. Query Info
print(f"EGL_VENDOR: {egl.eglQueryString(dpy, 0x3053)}")
print(f"EGL_VERSION: {egl.eglQueryString(dpy, 0x3054)}")
print(f"EGL_EXTENSIONS: {egl.eglQueryString(dpy, 0x3055)}")

# 4. Bind API (Try GLES)
print("Binding GLES API...")
if not egl.eglBindAPI(EGL_OPENGL_ES_API):
    print(f"eglBindAPI(GLES) failed: {hex(egl.eglGetError())}")
else:
    print("Bound GLES API")

# 5. Choose Config
config_attribs = [
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_BLUE_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_RED_SIZE, 8,
    EGL_DEPTH_SIZE, 24,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
    EGL_NONE
]
config_attribs_arr = (c_int * len(config_attribs))(*config_attribs)
configs = (c_void_p * 1)()
num_configs = c_int()

print("Choosing Config...")
if not egl.eglChooseConfig(dpy, config_attribs_arr, configs, 1, byref(num_configs)):
    print(f"eglChooseConfig failed: {hex(egl.eglGetError())}")
    sys.exit(1)

if num_configs.value == 0:
    print("No configs found!")
    sys.exit(1)
print(f"Found {num_configs.value} configs. Using first.")
config = configs[0]

# 6. Create Context
context_attribs = [
    EGL_CONTEXT_CLIENT_VERSION, 3,
    EGL_NONE
]
context_attribs_arr = (c_int * len(context_attribs))(*context_attribs)

print("Creating Context...")
ctx = egl.eglCreateContext(dpy, config, EGL_NO_CONTEXT, context_attribs_arr)
if not ctx:
    print(f"eglCreateContext failed: {hex(egl.eglGetError())}")
    sys.exit(1)
print(f"Context Created: {ctx}")

# 7. Make Current (Surfaceless)
print("Making Current (Surfaceless)...")
if not egl.eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx):
    print(f"eglMakeCurrent failed: {hex(egl.eglGetError())}")
    sys.exit(1)
print("Context Made Current!")

# 8. Query GL Version
try:
    print(f"Realpath of libGLESv2.so.2: {os.path.realpath('/usr/lib64/libGLESv2.so.2')}")
    
    nvidia_gles = "/usr/lib64/libGLESv2_nvidia.so.2"
    if os.path.exists(nvidia_gles):
        print(f"Found NVIDIA GLES lib: {nvidia_gles}")
        libgles_nv = CDLL(nvidia_gles, mode=RTLD_GLOBAL)
        libgles_nv.glGetString.argtypes = [c_uint]
        libgles_nv.glGetString.restype = c_char_p
        res_nv = libgles_nv.glGetString(0x1F02) # GL_VERSION
        print(f"NVIDIA GLES Direct Call: {res_nv}")
    else:
        print("NVIDIA GLES lib not found.")

    # Method A: Direct Call via Library
    print("--- Method A: Direct Library Call ---")
    libgles = CDLL("libGLESv2.so.2", mode=RTLD_GLOBAL)
    libgles.glGetString.argtypes = [c_uint]
    libgles.glGetString.restype = c_char_p
    GL_VERSION = 0x1F02
    
    res = libgles.glGetString(GL_VERSION)
    print(f"Direct glGetString(GL_VERSION): {res}")

    # Method B: eglGetProcAddress
    print("--- Method B: eglGetProcAddress ---")
    egl.eglGetProcAddress.argtypes = [c_char_p]
    egl.eglGetProcAddress.restype = c_void_p
    
    glGetString_ptr = egl.eglGetProcAddress(b"glGetString")
    print(f"eglGetProcAddress('glGetString'): {glGetString_ptr}")
    
    if glGetString_ptr:
        # Cast to function type
        GLGETSTRINGPROC = CFUNCTYPE(c_char_p, c_uint)
        glGetString = GLGETSTRINGPROC(glGetString_ptr)
        res_proc = glGetString(GL_VERSION)
        print(f"Proc glGetString(GL_VERSION): {res_proc}")
        
except Exception as e:
    print(f"Failed to query GL info: {e}")
