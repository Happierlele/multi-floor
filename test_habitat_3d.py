import os

# --- CRITICAL: Inject MESA Environment Variables BEFORE habitat_sim import ---
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"
os.environ["MESA_EXTENSION_OVERRIDE"] = "+GL_EXT_gpu_shader4"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"
# -------------------------------------------------------------------------

import habitat_sim
import numpy as np

def test_headless_rendering():
    print("Testing Habitat-Sim Headless Rendering...")
    
    # Check if we are in a headless environment
    display = os.environ.get("DISPLAY")
    print(f"DISPLAY env var: {display}")
    
    # Basic Configuration
    sim_cfg = habitat_sim.SimulatorConfiguration()
    # Use the same scene as in habitat_env.py
    sim_cfg.scene_id = "data/versioned_data/habitat_test_scenes/skokloster-castle.glb"
    
    # Disable physics to isolate rendering issues
    sim_cfg.enable_physics = False

    # Force GPU device ID to 0 (sometimes required for GLX context)
    sim_cfg.gpu_device_id = 0 
    
    # Not supported in this version of Habitat-Sim
    if hasattr(sim_cfg, 'gpu_gpu_transfer'):
        sim_cfg.gpu_gpu_transfer = False
    
    # Agent Configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    
    # Handle API differences between habitat-sim versions (0.1.7 vs new)
    SensorSpecClass = getattr(habitat_sim, "CameraSensorSpec", habitat_sim.SensorSpec)
    
    # RGB Sensor
    rgb_sensor_spec = SensorSpecClass()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [480, 640]
    rgb_sensor_spec.position = [0.0, 1.5, 0.0]
    
    agent_cfg.sensor_specifications = [rgb_sensor_spec]
    
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    try:
        print("Initializing Simulator...")
        sim = habitat_sim.Simulator(cfg)
        print("Simulator Initialized Successfully.")
        
        print("Initializing Agent...")
        agent = sim.initialize_agent(0)
        print("Agent Initialized.")
        
        print("rendering one frame...")
        observations = sim.get_sensor_observations()
        if "color_sensor" in observations:
            print("RGB Observation Shape:", observations["color_sensor"].shape)
            print("Rendering Test PASSED.")
        else:
            print("ERROR: color_sensor not found in observations.")
            
        sim.close()
        
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    test_headless_rendering()
