import os
import sys
import ctypes
import glob

# -------------------------------------------------------------------------
# CRITICAL FIX: Preload Configuration for Habitat-Sim (GLES + EGL + NVIDIA)
# -------------------------------------------------------------------------
# We need to preload:
# 1. Conda's libstdc++.so.6 (to satisfy PyTorch/LLVM)
# 2. System's libGLESv2_nvidia.so.2 (to bypass broken GLVND dispatcher)
preloads = []

# 1. Conda libstdc++
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    conda_stdc = os.path.join(conda_prefix, "lib", "libstdc++.so.6")
    if os.path.exists(conda_stdc):
        preloads.append(conda_stdc)
        print(f"[Driver] Found Conda libstdc++: {conda_stdc}", flush=True)

# 2. NVIDIA GLES lib
nvidia_gles_lib = "/usr/lib64/libGLESv2_nvidia.so.2"
if os.path.exists(nvidia_gles_lib):
    preloads.append(nvidia_gles_lib)
    print(f"[Driver] Found NVIDIA GLES lib: {nvidia_gles_lib}", flush=True)

# Apply LD_PRELOAD locally (for this process and forked workers)
current_preload = os.environ.get("LD_PRELOAD", "")
new_preloads = []
for p in preloads:
    if p not in current_preload:
        new_preloads.append(p)

if new_preloads:
    preload_str = ":".join(new_preloads)
    if current_preload:
        preload_str = f"{preload_str}:{current_preload}"
    
    print(f"[Driver] Setting LD_PRELOAD: {preload_str}", flush=True)
    os.environ["LD_PRELOAD"] = preload_str

    # Re-exec to apply LD_PRELOAD immediately to THIS process
    # This ensures that subsequent imports (like llvmlite/numba) use the preloaded libs
    print("[Driver] Re-executing driver.py to apply LD_PRELOAD...", flush=True)
    try:
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        print(f"[Driver] Failed to re-exec: {e}", flush=True)
        sys.exit(1)

# Set Habitat-Sim EGL/GLES Environment Variables Globally
os.environ["MAGNUM_TARGET_GLES"] = "1"
os.environ["MAGNUM_TARGET_HEADLESS"] = "1"
os.environ["MAGNUM_TARGET_EGL"] = "1"
os.environ["EGL_PLATFORM"] = "device"
print(f"[Driver] Set Env: GLES=1, HEADLESS=1, EGL=1, EGL_PLATFORM=device", flush=True)

# ----------------------------------------------------------------

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import numpy as np

import random
import sys

from model import PolicyNet, QNet, StairSwitchNet
from runner import RLRunner
from parameter_clean import *

print("Starting Ray init...", flush=True)
# Configure Ray to be offline/local as much as possible
# Limit object store memory to 2GB to prevent Vmem explosion on HPC
# Fix: AF_UNIX path length limit (107 bytes). Use /tmp/ray_{user} for short path.
import getpass
user = getpass.getuser()
ray_tmp_dir = f"/tmp/ray_{user}_hab"
if not os.path.exists(ray_tmp_dir):
    try:
        os.makedirs(ray_tmp_dir)
    except OSError:
        # Fallback if /tmp is not writable
        ray_tmp_dir = os.path.join(os.getcwd(), "tmp")
        if not os.path.exists(ray_tmp_dir):
            os.makedirs(ray_tmp_dir)

print(f"[Driver] Using Ray temp dir: {ray_tmp_dir}", flush=True)

# Explicitly detect resources to inform Ray
num_gpus_available = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"[Driver] Detected {num_gpus_available} GPUs via torch", flush=True)

# Do NOT force SW render anymore, let it fail over to GPU EGL
if "FORCE_SW_RENDER" in os.environ:
    del os.environ["FORCE_SW_RENDER"]

ray.init(
    include_dashboard=False, 
    _temp_dir=ray_tmp_dir, 
    object_store_memory=2*1024*1024*1024,
    num_cpus=2, # Restrict CPU usage to prevent oversubscription on login nodes
    num_gpus=num_gpus_available, # Explicitly tell Ray how many GPUs we have
    runtime_env={
        "env_vars": {
            "LD_PRELOAD": os.environ.get("LD_PRELOAD", ""),
            "MAGNUM_TARGET_GLES": "1",
            "MAGNUM_TARGET_HEADLESS": "1",
            "MAGNUM_TARGET_EGL": "1",
            "EGL_PLATFORM": "surfaceless",
            # Ensure NVIDIA driver path is set
            "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
        }
    }
)
print("Ray init done.", flush=True)
print("Welcome to RL autonomous exploration!", flush=True)

writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


def main():
    # use GPU/CPU for driver/worker
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print(f"Driver running on: {device}", flush=True)
    
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    # initialize neural networks
    global_policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net1 = QNet(NODE_INPUT_DIM + 1, EMBEDDING_DIM).to(device)
    global_q_net2 = QNet(NODE_INPUT_DIM + 1, EMBEDDING_DIM).to(device)
    log_alpha = torch.FloatTensor([-2]).to(device)
    log_alpha.requires_grad = True

    global_stair_switch_net = StairSwitchNet().to(device)
    global_target_stair_switch_net = StairSwitchNet().to(device)

    global_target_q_net1 = QNet(NODE_INPUT_DIM + 1, EMBEDDING_DIM).to(device)
    global_target_q_net2 = QNet(NODE_INPUT_DIM + 1, EMBEDDING_DIM).to(device)
    
    # initialize optimizers
    global_policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=LR)
    global_q_net1_optimizer = optim.Adam(global_q_net1.parameters(), lr=LR)
    global_q_net2_optimizer = optim.Adam(global_q_net2.parameters(), lr=LR)
    log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)
    stair_switch_optimizer = optim.Adam(global_stair_switch_net.parameters(), lr=1e-4)

    # target entropy for SAC
    entropy_target = 0.05 * (-np.log(1 / K_SIZE))

    curr_episode = 0
    target_q_update_counter = 1

    # load model and optimizer trained before
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth', map_location=device)
        global_policy_net.load_state_dict(checkpoint['policy_model'])
        global_q_net1.load_state_dict(checkpoint['q_net1_model'])
        global_q_net2.load_state_dict(checkpoint['q_net2_model'])
        log_alpha = checkpoint['log_alpha']
        log_alpha = checkpoint['log_alpha']
        log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)

        global_policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        global_q_net1_optimizer.load_state_dict(checkpoint['q_net1_optimizer'])
        global_q_net2_optimizer.load_state_dict(checkpoint['q_net2_optimizer'])
        log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
        curr_episode = checkpoint['episode']

        print("curr_episode set to ", curr_episode)
        print(log_alpha, log_alpha.requires_grad)
        print(global_policy_optimizer.state_dict()['param_groups'][0]['lr'])

    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()

    # launch meta agents
    print("Starting runners...", flush=True)
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]
    print("Runners started.", flush=True)

    # get global networks weights
    global_weights = global_policy_net.state_dict()
    global_switch_weights = global_stair_switch_net.state_dict()
    
    # Move weights to CPU before sending to workers to avoid CUDA serialization issues
    # Ray workers might not have CUDA visibility initialized when deserializing
    global_weights_cpu = {k: v.cpu() for k, v in global_weights.items()}
    global_switch_weights_cpu = {k: v.cpu() for k, v in global_switch_weights.items()}

    weights_set = [global_weights_cpu, global_switch_weights_cpu]
    
    # Serialize to bytes to ensure safe transfer and allow map_location='cpu' on worker
    import io
    weights_buffer = io.BytesIO()
    torch.save(weights_set, weights_buffer)
    weights_buffer.seek(0)
    weights_bytes = weights_buffer.read()

    # Initialize workers (this triggers the EGL initialization in remote workers)
    # We do a dummy call to ensure they are alive before proceeding
    print("Initializing workers...", flush=True)
    ray.get([agent.job.remote(weights_bytes, 0) for agent in meta_agents])
    print("Workers initialized.", flush=True)

    # distributed training if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DataParallel")
        dp_policy = nn.DataParallel(global_policy_net)
        dp_q_net1 = nn.DataParallel(global_q_net1)
        dp_q_net2 = nn.DataParallel(global_q_net2)
        dp_target_q_net1 = nn.DataParallel(global_target_q_net1)
        dp_target_q_net2 = nn.DataParallel(global_target_q_net2)
    else:
        print("Using single device (GPU/CPU) - Skipping DataParallel")
        dp_policy = global_policy_net
        dp_q_net1 = global_q_net1
        dp_q_net2 = global_q_net2
        dp_target_q_net1 = global_target_q_net1
        dp_target_q_net2 = global_target_q_net2

    # launch the first job on each runner
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote(weights_bytes, curr_episode))

    # initialize metric collector
    metric_name = ['travel_dist', 'success_rate', 'explored_rate']
    training_data = []
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []

    # initialize training replay buffer
    experience_buffer = []
    for i in range(27):
        experience_buffer.append([])
    
    stair_switch_buffer = []
    for i in range(5):
        stair_switch_buffer.append([])

    # collect data from worker and do training
    try:
        while True:
            # wait for any job to be completed
            done_id, job_list = ray.wait(job_list)
            # get the results
            done_jobs = ray.get(done_id)

            # save experience and metric
            for job in done_jobs:
                job_results, metrics, info = job
                for i in range(len(experience_buffer)):
                    experience_buffer[i] += job_results[i]
                
                # Collect StairSwitchNet data
                if len(job_results) > 27:
                    ss_data = job_results[27]
                    if ss_data: # ss_data is [s, a, r, s', d] tensors
                        for k in range(5):
                            stair_switch_buffer[k] += list(ss_data[k])

                for n in metric_name:
                    perf_metrics[n].append(metrics[n])
            buffer_size = len(experience_buffer[0])
            if buffer_size == 0:
                print("Warning: Experience buffer is empty. Waiting for more data...")
                # Re-submit the job if it failed to produce data (optional, but good for liveness)
                # But here we just continue to wait for other jobs
                if len(job_list) == 0 and curr_episode < 10000: # Ensure we don't get stuck if all jobs fail
                     # Restart a job? Logic is complex here, usually job_list is kept full.
                     # But for now, just continue is safer than crashing.
                     pass
            
            # Robust check for buffer consistency
            valid_buffer = True
            for i in range(len(experience_buffer)):
                if len(experience_buffer[i]) != buffer_size:
                    print(f"Error: Experience buffer mismatch at index {i}. Expected {buffer_size}, got {len(experience_buffer[i])}")
                    valid_buffer = False
                    break
            
            if not valid_buffer:
                print("Clearing inconsistent experience buffer to prevent crash.")
                for i in range(len(experience_buffer)):
                    experience_buffer[i] = []
                continue

            # launch new task
            curr_episode += 1
            # Re-serialize updated weights
            global_weights = global_policy_net.state_dict()
            global_switch_weights = global_stair_switch_net.state_dict()
            global_weights_cpu = {k: v.cpu() for k, v in global_weights.items()}
            global_switch_weights_cpu = {k: v.cpu() for k, v in global_switch_weights.items()}
            weights_set = [global_weights_cpu, global_switch_weights_cpu]
            
            weights_buffer = io.BytesIO()
            torch.save(weights_set, weights_buffer)
            weights_buffer.seek(0)
            weights_bytes = weights_buffer.read()
            
            job_list.append(meta_agents[info['id']].job.remote(weights_bytes, curr_episode))

            # start training
            if curr_episode % 1 == 0 and len(experience_buffer[0]) >= MINIMUM_BUFFER_SIZE:
                print("training")

                # keep the replay buffer size
                if len(experience_buffer[0]) >= REPLAY_SIZE:
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]
                
                if len(stair_switch_buffer[0]) >= REPLAY_SIZE:
                    for i in range(5):
                        stair_switch_buffer[i] = stair_switch_buffer[i][-REPLAY_SIZE:]

                indices = range(len(experience_buffer[0]))

                # training for n times each step
                for j in range(8):
                    # randomly sample a batch data
                    sample_indices = random.sample(indices, BATCH_SIZE)
                    rollouts = []
                    for i in range(len(experience_buffer)):
                        rollouts.append([experience_buffer[i][index] for index in sample_indices])

                    # stack batch data to tensors
                    node_inputs = torch.stack(rollouts[0]).to(device)
                    node_padding_mask = torch.stack(rollouts[1]).to(device)
                    edge_mask = torch.stack(rollouts[2]).to(device)
                    current_index = torch.stack(rollouts[3]).to(device)
                    current_edge = torch.stack(rollouts[4]).to(device)
                    edge_padding_mask = torch.stack(rollouts[5]).to(device)
                    action = torch.stack(rollouts[6]).to(device)
                    reward = torch.stack(rollouts[7]).to(device)
                    done = torch.stack(rollouts[8]).to(device)
                    next_node_inputs = torch.stack(rollouts[9]).to(device)
                    next_node_padding_mask = torch.stack(rollouts[10]).to(device)
                    next_edge_mask = torch.stack(rollouts[11]).to(device)
                    next_current_index = torch.stack(rollouts[12]).to(device)
                    next_current_edge = torch.stack(rollouts[13]).to(device)
                    next_edge_padding_mask = torch.stack(rollouts[14]).to(device)

                    critic_node_inputs = torch.stack(rollouts[15]).to(device)
                    critic_node_padding_mask = torch.stack(rollouts[16]).to(device)
                    critic_edge_mask = torch.stack(rollouts[17]).to(device)
                    critic_current_index = torch.stack(rollouts[18]).to(device)
                    critic_current_edge = torch.stack(rollouts[19]).to(device)
                    critic_edge_padding_mask = torch.stack(rollouts[20]).to(device)
                    critic_next_node_inputs = torch.stack(rollouts[21]).to(device)
                    critic_next_node_padding_mask = torch.stack(rollouts[22]).to(device)
                    critic_next_edge_mask = torch.stack(rollouts[23]).to(device)
                    critic_next_current_index = torch.stack(rollouts[24]).to(device)
                    critic_next_current_edge = torch.stack(rollouts[25]).to(device)
                    critic_next_edge_padding_mask = torch.stack(rollouts[26]).to(device)

                    observation = [node_inputs, node_padding_mask, edge_mask, current_index,
                                   current_edge, edge_padding_mask]
                    next_observation = [next_node_inputs, next_node_padding_mask, next_edge_mask,
                                        next_current_index, next_current_edge, next_edge_padding_mask]

                    critic_observation = [critic_node_inputs, critic_node_padding_mask, critic_edge_mask,
                                          critic_current_index,
                                          critic_current_edge, critic_edge_padding_mask]
                    critic_next_observation = [critic_next_node_inputs, critic_next_node_padding_mask,
                                               critic_next_edge_mask,
                                               critic_next_current_index, critic_next_current_edge,
                                               critic_next_edge_padding_mask]

                    # SAC
                    with torch.no_grad():
                        q_values1 = dp_q_net1(*critic_observation)
                        q_values2 = dp_q_net2(*critic_observation)
                        q_values = torch.min(q_values1, q_values2)

                    logp = dp_policy(*observation)
                    policy_loss = torch.sum(
                        (logp.exp().unsqueeze(2) * (log_alpha.exp().detach() * logp.unsqueeze(2) - q_values.detach())),
                        dim=1).mean()

                    global_policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), max_norm=100,
                                                                      norm_type=2)
                    global_policy_optimizer.step()

                    with torch.no_grad():
                        next_logp = dp_policy(*next_observation)
                        next_q_values1 = dp_target_q_net1(*critic_next_observation)
                        next_q_values2 = dp_target_q_net2(*critic_next_observation)
                        next_q_values = torch.min(next_q_values1, next_q_values2)
                        value_prime = torch.sum(
                            next_logp.unsqueeze(2).exp() * (next_q_values - log_alpha.exp() * next_logp.unsqueeze(2)),
                            dim=1).unsqueeze(1)
                        target_q = reward + GAMMA * (1 - done) * value_prime

                    mse_loss = nn.MSELoss()

                    q_values1 = dp_q_net1(*critic_observation)
                    q1 = torch.gather(q_values1, 1, action)
                    q1_loss = mse_loss(q1, target_q.detach()).mean()

                    global_q_net1_optimizer.zero_grad()
                    q1_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net1.parameters(), max_norm=20000,
                                                                 norm_type=2)
                    global_q_net1_optimizer.step()

                    q_values2 = dp_q_net2(*critic_observation)
                    q2 = torch.gather(q_values2, 1, action)
                    q2_loss = mse_loss(q2, target_q.detach()).mean()

                    global_q_net2_optimizer.zero_grad()
                    q2_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net2.parameters(), max_norm=20000,
                                                                 norm_type=2)
                    global_q_net2_optimizer.step()

                    entropy = (logp * logp.exp()).sum(dim=-1)
                    alpha_loss = -(log_alpha * (entropy.detach() + entropy_target)).mean()

                    log_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    log_alpha_optimizer.step()

                    # StairSwitchNet Training
                    if len(stair_switch_buffer[0]) >= BATCH_SIZE:
                        ss_indices = range(len(stair_switch_buffer[0]))
                        ss_sample_indices = random.sample(ss_indices, BATCH_SIZE)
                        
                        ss_batch = []
                        for k in range(5):
                            ss_batch.append(torch.stack([stair_switch_buffer[k][idx] for idx in ss_sample_indices]).to(device))
                        
                        s, a, r, next_s, d = ss_batch
                        
                        # Q-learning loss
                        q_preds = global_stair_switch_net(s)
                        q_action = q_preds.gather(1, a.long())
                        
                        with torch.no_grad():
                            q_next = global_target_stair_switch_net(next_s)
                            q_next_max = q_next.max(1)[0].unsqueeze(1)
                            target = r + GAMMA * (1 - d) * q_next_max
                        
                        ss_loss = nn.MSELoss()(q_action, target)
                        
                        stair_switch_optimizer.zero_grad()
                        ss_loss.backward()
                        stair_switch_optimizer.step()
                    else:
                         ss_loss = torch.tensor(0.0)

                    target_q_update_counter += 1
                    # print("target q update counter", target_q_update_counter % 1024)

                # data record to be written in tensorboard
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward.mean().item(), value_prime.mean().item(), policy_loss.item(), q1_loss.item(),
                        entropy.mean().item(), policy_grad_norm.item(), q_grad_norm.item(), log_alpha.item(),
                        alpha_loss.item(), ss_loss.item(), *perf_data]
                training_data.append(data)

            # write record to tensorboard
            if len(training_data) >= SUMMARY_WINDOW:
                write_to_tensor_board(writer, training_data, curr_episode)
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []

            # get the updated global weights
            weights_set = []
            if device != local_device:
                policy_weights = global_policy_net.to(local_device).state_dict()
                switch_weights = global_stair_switch_net.to(local_device).state_dict()
                global_policy_net.to(device)
                global_stair_switch_net.to(device)
            else:
                policy_weights = global_policy_net.state_dict()
                switch_weights = global_stair_switch_net.state_dict()
            weights_set.append(policy_weights)
            weights_set.append(switch_weights)

            # update the target q net
            if target_q_update_counter > 64:
                print("update target q net")
                target_q_update_counter = 1
                global_target_q_net1.load_state_dict(global_q_net1.state_dict())
                global_target_q_net2.load_state_dict(global_q_net2.state_dict())
                global_target_q_net1.eval()
                global_target_q_net2.eval()
                
                global_target_stair_switch_net.load_state_dict(global_stair_switch_net.state_dict())
                global_target_stair_switch_net.eval()

            # save the model
            if curr_episode % 32 == 0:
                print('Saving model', end='\n')
                checkpoint = {"policy_model": global_policy_net.state_dict(),
                              "q_net1_model": global_q_net1.state_dict(),
                              "q_net2_model": global_q_net2.state_dict(),
                              "log_alpha": log_alpha,
                              "policy_optimizer": global_policy_optimizer.state_dict(),
                              "q_net1_optimizer": global_q_net1_optimizer.state_dict(),
                              "q_net2_optimizer": global_q_net2_optimizer.state_dict(),
                              "log_alpha_optimizer": log_alpha_optimizer.state_dict(),
                              "stair_switch_model": global_stair_switch_net.state_dict(),
                              "stair_switch_optimizer": stair_switch_optimizer.state_dict(),
                              "episode": curr_episode,
                              }
                path_checkpoint = "./" + model_path + "/checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


def write_to_tensor_board(writer, tensorboard_data, curr_episode):
    # each row in tensorboardData represents an episode
    # each column is a specific metric

    tensorboard_data = np.array(tensorboard_data)
    tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
    reward, value, policy_loss, q_value_loss, entropy, policy_grad_norm, q_value_grad_norm, log_alpha, alpha_loss, ss_loss, travel_dist, success_rate, explored_rate = tensorboard_data

    writer.add_scalar(tag='Losses/Value', scalar_value=value, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Loss', scalar_value=policy_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Alpha Loss', scalar_value=alpha_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Loss', scalar_value=q_value_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Entropy', scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Grad Norm', scalar_value=policy_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Grad Norm', scalar_value=q_value_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Log Alpha', scalar_value=log_alpha, global_step=curr_episode)
    writer.add_scalar(tag='Losses/StairSwitch Loss', scalar_value=ss_loss, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Reward', scalar_value=reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Travel Distance', scalar_value=travel_dist, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Explored Rate', scalar_value=explored_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Success Rate', scalar_value=success_rate, global_step=curr_episode)


if __name__ == "__main__":
    main()
