from parameter_clean import *

# CRITICAL: Import Worker (and thus configure environment) BEFORE importing torch
# This ensures that EGL/GL environment variables and library preloading happen 
# before Torch initializes CUDA or loads conflicting libraries.
if USE_HABITAT:
    from worker_habitat import Worker
else:
    from worker import Worker

import torch
import ray
from model import PolicyNet, StairSwitchNet


class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        # Dynamically check for CUDA availability. 
        # Even if USE_GPU=True, if no GPU is found (e.g. Ray CPU node), fallback to CPU.
        self.device = torch.device('cuda') if USE_GPU and torch.cuda.is_available() else torch.device('cpu')
        print(f"[Runner {meta_agent_id}] Initialized on device: {self.device}", flush=True)
        self.network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
        self.network.to(self.device)
        self.stair_switch_net = StairSwitchNet().to(self.device)
        self.worker = None

    def get_weights(self):
        return self.network.state_dict(), self.stair_switch_net.state_dict()

    def set_weights(self, policy_weights, switch_weights):
        self.network.load_state_dict(policy_weights)
        if switch_weights is not None:
            self.stair_switch_net.load_state_dict(switch_weights)

    def do_job(self, episode_number):
        save_img = True if episode_number % SAVE_IMG_GAP == 0 else False
        
        # Reuse worker if available (Persistent Worker Pattern)
        if self.worker is None:
            self.worker = Worker(self.meta_agent_id, self.network, self.stair_switch_net, episode_number, device=self.device, save_image=save_img,
                        use_vlm=USE_VLM, vlm_model_name=VLM_MODEL_NAME)
        else:
            # If worker exists (e.g. initialized in RLRunner), reset it for new episode
            # Check if reset_episode method exists (it should in worker_habitat)
            if hasattr(self.worker, 'reset_episode'):
                self.worker.reset_episode(episode_number, save_img)
            else:
                # Fallback for non-habitat worker or if reset not implemented
                # We might have to recreate it if it doesn't support reset
                # But for now assuming habitat worker supports it.
                pass 

        self.worker.run_episode()

        job_results = self.worker.episode_buffer
        perf_metrics = self.worker.perf_metrics
        return job_results, perf_metrics

    def job(self, weights_bytes, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        
        # Deserialize weights from bytes with map_location='cpu'
        # This bypasses Ray's auto-deserialization and ensures we don't try to load onto a missing CUDA device
        import io
        weights_buffer = io.BytesIO(weights_bytes)
        weights_set = torch.load(weights_buffer, map_location='cpu')

        # set the local weights to the global weight values from the master network
        # Move weights to CPU before loading to avoid CUDA deserialization error on workers without GPU visibility
        policy_weights = {k: v.cpu() for k, v in weights_set[0].items()}
        switch_weights = {k: v.cpu() for k, v in weights_set[1].items()} if len(weights_set) > 1 and weights_set[1] is not None else None
        
        self.set_weights(policy_weights, switch_weights)
        
        # Ensure model is on correct device after loading weights
        self.network.to(self.device)
        self.stair_switch_net.to(self.device)

        job_results, metrics = self.do_job(episode_number)
        
        # Move results to CPU before returning to avoid CUDA deserialization error on driver
        # job_results is a list of lists of tensors
        cpu_job_results = []
        for buffer in job_results:
            cpu_buffer = []
            for item in buffer:
                if isinstance(item, torch.Tensor):
                    cpu_buffer.append(item.cpu())
                else:
                    cpu_buffer.append(item)
            cpu_job_results.append(cpu_buffer)
            
        info = {"id": self.meta_agent_id, "episode_number": episode_number}

        return cpu_job_results, metrics, info


# Fix: Check CUDA availability dynamically for Ray resources WITHOUT initializing Torch
# Using nvidia-smi check to avoid premature CUDA initialization which conflicts with Habitat-Sim EGL
try:
    import subprocess
    subprocess.check_call(['nvidia-smi', '-L'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    driver_has_gpu = True
except:
    driver_has_gpu = False

@ray.remote(num_cpus=1, num_gpus=(NUM_GPU / NUM_META_AGENT) if driver_has_gpu else 0)
class RLRunner(Runner):
    def __init__(self, meta_agent_id):
        # ---------------------------------------------------------------------
        # CRITICAL: Initialize Habitat-Sim (EGL) BEFORE Torch CUDA
        # ---------------------------------------------------------------------
        # If we let Torch initialize CUDA first, EGL might fail to find devices 
        # or crash with GL::Renderer::Error::InvalidValue.
        # We initialize the HabitatEnv HERE and persist it.
        
        self.persistent_env = None
        if USE_HABITAT:
            try:
                print(f"[RLRunner {meta_agent_id}] Initializing Persistent HabitatEnv...", flush=True)
                from habitat_env import HabitatEnv
                # Initialize with dummy values, will be reset later via reset_episode
                # This ensures EGL context is created and claimed before Torch
                self.persistent_env = HabitatEnv(0, plot=True)
                print(f"[RLRunner {meta_agent_id}] Persistent HabitatEnv initialized!", flush=True)
            except Exception as e:
                print(f"[RLRunner {meta_agent_id}] Failed to init HabitatEnv: {e}", flush=True)
                # We raise exception here because if env fails, this worker is useless
                raise e
        
        # Now proceed with standard Runner initialization (which initializes Torch)
        super().__init__(meta_agent_id)
        
        # Create persistent Worker using the pre-initialized environment
        if self.persistent_env:
            print(f"[RLRunner {meta_agent_id}] Creating Persistent Worker...", flush=True)
            self.worker = Worker(self.meta_agent_id, self.network, self.stair_switch_net, 0, device=self.device, save_image=True,
                        use_vlm=USE_VLM, vlm_model_name=VLM_MODEL_NAME, env=self.persistent_env)
            print(f"[RLRunner {meta_agent_id}] Persistent Worker created!", flush=True)


if __name__ == '__main__':
    ray.init()
    runner = RLRunner.remote(0)
    job_id = runner.do_job.remote(1)
    out = ray.get(job_id)
    print(out[1])
