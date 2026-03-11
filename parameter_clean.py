# saving path
FOLDER_NAME = 'habitat_test_v1'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'

# save training data
SUMMARY_WINDOW = 1 # Write to TensorBoard every single episode (Real-time updates!)
LOAD_MODEL = False
SAVE_IMG_GAP = 1

# map and planning resolution
CELL_SIZE = 0.4
# Use finer node resolution to increase neighbor density in cluttered MP3D scenes
NODE_RESOLUTION = 2.0
FRONTIER_CELL_SIZE = 2 * CELL_SIZE

# map representation
FREE = 255
OCCUPIED = 1
UNKNOWN = 127

# sensor and utility range
SENSOR_RANGE = 16
UTILITY_RANGE = 0.8 * SENSOR_RANGE
MIN_UTILITY = 2

# updating map range w.r.t the robot
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION

# training parameters
# Increase episode horizon to allow finishing one floor in a single run
MAX_EPISODE_STEP = 1024
REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 1000 # Reduced to match BATCH_SIZE to trigger training immediately after one episode
BATCH_SIZE = 128
LR = 1e-5
GAMMA = 1
NUM_META_AGENT = 1

# network parameters
NODE_INPUT_DIM = 5 # Updated to 5 to match GroundTruthNodeManager output (coords(2) + util(1) + expl(1) + visit(1))
EMBEDDING_DIM = 128

# Graph parameters
K_SIZE = 25
NODE_PADDING_SIZE = 1024

# GPU usage
USE_GPU = True
USE_GPU_GLOBAL = True
NUM_GPU = 1

# VLM parameters
USE_VLM = True
VLM_MODEL_NAME = "qwen-vl-max"

# Environment Type
USE_HABITAT = True

# stairs and floor switching
STAIRS_VALUE = 180
FLOOR_SWITCH_EXP_RATE = 0.8
STAIRS_DIST_TOLERANCE = 2.0 # Reduced from NODE_RESOLUTION * 1.5 (6.0m) to 2.0m to prevent early triggering
STAIR_INTERNAL_STEPS = 60
INTER_FLOOR_TIMEOUT_STEPS = 200
MAX_STAIRS_PER_MAP = 3
STAIRS_NEXT_FLOOR_RADIUS = 10
STAIR_DIST_WEIGHT = 1.0
STAIR_NEXT_AREA_WEIGHT = 0.01

STUCK_WINDOW = 15
STUCK_UNIQUE_POS_THRESHOLD = 4

# utility based switching
UTILITY_SWITCH_THRESHOLD = 5.0  # If max utility is below this
MIN_EXP_RATE_FOR_UTILITY_SWITCH = 0.4  # And exploration rate is above this, go to stairs
