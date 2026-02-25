# large-scale-DRL-exploration
[RAL 2024] Deep Reinforcement Learning-based Large-scale Robot Exploration - - Public code and model

**Note**: This is a new implementation of ARiADNE ground truth critic variant. 
You can find our original implementation in the [main branch](https://github.com/marmotlab/large-scale-DRL-exploration/tree/main).
We reimplement the code to optimize the computing time, RAM/VRAM usage, and compatibility with ROS. 
The trained model can be directly tested in our [ARiADNE ROS planner](https://github.com/marmotlab/ARiADNE-ROS-Planner).

## Run

#### Dependencies
We recommend to use conda for package management. 
Our planner is coded in Python and based on Pytorch. 
Other than Pytorch, please install following packages by:
```
pip install scikit-image matplotlib ray tensorboard
```
We tested our planner in various version of these packages so you can just install the latest one.

#### Training
Download this repo and go into the folder:
```
git clone https://github.com/marmotlab/large-scale-DRL-exploration
cd ARiADNE
```
Launch your conda environment if any and run:

```python driver.py```

The default training code requires around 8GB VRAM and 20G RAM. 
You can modify the hyperparameters in `parameter.py`.


## Files
* `parameters.py` Training parameters.
* `driver.py` Driver of training program, maintain & update the global network.
* `runner.py` Wrapper of the workers.
* `worker.py` Interact with environment and collect episode experience.
* `model.py` Define attention-based network.
* `env.py` Autonomous exploration environment.
* `node_manager.py` Manage and update the informative graph for policy observation.
* `ground_truth_node_manager.py` Manage and update the ground truth informative graph for critic observation.
* `quads` Quad tree for node indexing provided by [Daniel Lindsley](https://github.com/toastdriven).
* `sensor.py` Simulate the sensor model of Lidar.
* `utils` Some helper functions.
* `/maps` Maps of training environments provided by <a href="https://github.com/RobustFieldAutonomyLab/DRL_robot_exploration">Chen et al.</a>.


### Authors
[Yuhong Cao](https://github.com/caoyuhong001)\
Rui Zhao\
[Yizhuo Wang](https://github.com/wyzh98)\
Bairan Xiang\
[Guillaume Sartoretti](https://github.com/gsartoretti)
