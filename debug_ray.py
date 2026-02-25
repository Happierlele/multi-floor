import ray
import time
import os

print(f"Testing Ray on node: {os.uname().nodename}")
print("1. Starting ray.init()...")
start = time.time()

# 使用与 driver.py 相同的配置进行测试
try:
    ray.init(
        include_dashboard=False, 
        _temp_dir=os.environ.get("RAY_TMPDIR", None),
        object_store_memory=512*1024*1024 # 使用较小的内存测试
    )
    print(f"2. Ray init success! Time taken: {time.time() - start:.2f}s")
except Exception as e:
    print(f"2. Ray init FAILED: {e}")
    exit(1)

@ray.remote
def ping():
    return "pong"

print("3. Testing remote task execution...")
try:
    ref = ping.remote()
    res = ray.get(ref, timeout=10.0)
    print(f"4. Task success! Result: {res}")
except Exception as e:
    print(f"4. Task FAILED: {e}")

print("5. Shutting down...")
ray.shutdown()
print("Done.")
