
import os

def _mask_key(k):
    if not k:
        return None
    k = str(k)
    if len(k) <= 12:
        return k[:2] + "***" + k[-2:]
    return k[:6] + "..." + k[-4:]

print("=== Basic ===")
print("python:", os.sys.version.replace("\n", " "))
print("cwd:", os.getcwd())

print("\n=== parameter_clean ===")
try:
    import parameter_clean as p
    print("import: OK")
    print("USE_VLM:", getattr(p, "USE_VLM", None))
    print("VLM_MODEL_NAME:", getattr(p, "VLM_MODEL_NAME", None))
    print("gifs_path:", getattr(p, "gifs_path", None))
except Exception as e:
    print("import: FAILED:", repr(e))
    p = None

print("\n=== Env Vars (VLM) ===")
qwen_key = os.getenv("QWEN_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
qwen_base = os.getenv("QWEN_BASE_URL")
openai_base = os.getenv("OPENAI_BASE_URL")
print("QWEN_API_KEY:", _mask_key(qwen_key))
print("OPENAI_API_KEY:", _mask_key(openai_key))
print("QWEN_BASE_URL:", qwen_base)
print("OPENAI_BASE_URL:", openai_base)

print("\n=== openai library ===")
try:
    import openai
    ver = getattr(openai, "__version__", None)
    print("import: OK", "(version:", ver, ")")
except Exception as e:
    print("import: FAILED:", repr(e))

print("\n=== VLMAdapter init ===")
try:
    from vlm_adapter import VLMAdapter
    model_name = "qwen-vl-max"
    if p is not None and getattr(p, "VLM_MODEL_NAME", None):
        model_name = str(getattr(p, "VLM_MODEL_NAME"))

    key = qwen_key or openai_key
    base = qwen_base or openai_base
    va = VLMAdapter(model_name=model_name, api_key=key, base_url=base)
    print("provider:", getattr(va, "provider", None))
    print("model_name:", getattr(va, "model_name", None))
    print("base_url:", getattr(va, "base_url", None))
    print("client_ready:", bool(getattr(va, "client", None)) or bool(getattr(va, "use_legacy_openai", False)) or bool(getattr(va, "http_fallback", False)))
except Exception as e:
    print("init: FAILED:", repr(e))
    va = None

print("\n=== Verdict (matches worker_habitat behavior) ===")
want_vlm = True if (p is None) else bool(getattr(p, "USE_VLM", False))
has_key = bool(qwen_key) or bool(openai_key)
adapter_ready = bool(va) and (bool(getattr(va, "client", None)) or bool(getattr(va, "use_legacy_openai", False)) or bool(getattr(va, "http_fallback", False)))
enabled = bool(want_vlm and has_key and adapter_ready)
print("want_vlm:", want_vlm)
print("has_key:", has_key)
print("adapter_ready:", adapter_ready)
print("VLM_enabled:", enabled)

print("\nTip: run `python test_imports.py` on your HPC login node to see exactly why VLM is disabled.")
