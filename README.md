# DQN Robot Navigation with IR-SIM

A hobby implementation of Deep Q-Network (DQN) for autonomous robot navigation in maze environments. The robot learns to navigate from random start positions to goal locations using LiDAR sensors and position information.

> **Inspired by:** "Federated Reinforcement Learning Framework for Mobile Robot Navigation Using ROS and Gazebo" by Xing An et al. (IEEE Internet of Things Magazine, 2025). While that paper tackles federated learning with multiple robots using ROS/Gazebo, this is a simplified Non-Federated Learning single robot implementation using IR-SIM - perfect for learning and experimentation.

## Demo

https://github.com/user-attachments/assets/6dafb664-7841-4e3c-bddc-c54f0255b94a

## Features

- Gymnasium-compatible environment wrapper for IR-SIM
- IR-SIM custom behavior wrapper for Gymnasium control integration
- DQN with experience replay and target network
- Discrete action space with 7 actions
- LiDAR-based perception (24 laser beams)
- Randomized training environments for better generalization
- Checkpoint saving and model evaluation
- Training visualization and metrics plotting
- Docker support for easy setup

## Installation

**Tested on:** Windows 11 with Docker Desktop installed (NVIDIA GPU Required)

### Option 1: Docker (Recommended)

```bash
# Build the image
docker build -t maze-rl .

# Run with GPU support
docker run --gpus all -it -v ${PWD}:/app maze-rl
```

### Option 2: Manual Installation (Ubuntu 24.04)

```bash
# Install IR-SIM
pip3 install ir-sim
pip3 install ir-sim[all]

# Install PyTorch (adjust for your CUDA version)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# Install other dependencies
pip3 install gymnasium pygame tqdm

# Install system dependencies (Ubuntu/Debian)
apt install python3-pyqt5 python3-tk
```

## Project Structure

```
.
├── custom_behavior.py      # Custom behavior for Gymnasium control
├── irsim_env.py            # Gymnasium environment wrapper
├── train_dqn_irsim.py      # DQN training script
├── test_dqn_irsim.py       # Model testing script
├── world/                  # World configuration files
│   ├── irsim_world.yaml
|   └── ...
├── checkpoint/             # Saved models and checkpoints
└── README.md
```

## Usage

### Training

Train a DQN agent with default settings:

```bash
python train_dqn_irsim.py --episodes 1000
```

**Available training options:**
```bash
python train_dqn_irsim.py \
    --episodes 1000 \              # Number of training episodes
    --render \                      # Enable visualization (slower)
    --no-randomize \                # Use fixed start/goal positions
    --world irsim_world.yaml        # World configuration file
```

Training automatically:
- Saves checkpoints every 100 episodes to `checkpoint/checkpoint_<episode>.pth`
- Saves the final model to `checkpoint/dqn_final.pth`
- Generates training plots at `checkpoint/training_results.png`

### Testing

Evaluate a trained model:

```bash
python test_dqn_irsim.py --checkpoint dqn_final.pth --episodes 10
```

**Available testing options:**
```bash
python test_dqn_irsim.py \
    --checkpoint dqn_final.pth \    # Model checkpoint to load
    --episodes 10 \                 # Number of test episodes
    --no-render \                   # Run without visualization
    --no-randomize \                # Use fixed positions
    --world irsim_world.yaml \      # World configuration file
    --seed 42                       # Random seed for reproducibility
```

## Environment Details

### Observation Space

The observation is a dictionary containing:
- **`robot_location`**: Current position and orientation [x, y, θ]
- **`goal_location`**: Target position [x, y, θ]
- **`lidar`**: 24 laser range measurements

This is flattened into a 28-dimensional state vector:
- 24 normalized LiDAR readings (0-1 range)
- Distance to goal (normalized)
- Angle to goal (normalized)
- Distance to nearest obstacle (normalized)
- Angle to nearest obstacle (normalized)

### Action Space

7 discrete actions with constant linear velocity (0.5 m/s) and varying angular velocities:
1. Strong right turn (-1.0 rad/s)
2. Medium right turn (-0.66 rad/s)
3. Weak right turn (-0.33 rad/s)
4. Straight (0.0 rad/s)
5. Weak left turn (+0.33 rad/s)
6. Medium left turn (+0.66 rad/s)
7. Strong left turn (+1.0 rad/s)

### Reward Function

Multi-component reward designed to encourage efficient navigation:
- **Goal reached**: +100
- **Collision**: -100
- **Angle reward** (R_θ): Encourages facing toward the goal
- **Distance reward** (R_d): Encourages making progress toward the goal
- **Obstacle penalty**: Penalty for being too close to obstacles

The combined reward: `R = R_θ × R_d + R_obstacle`

## Configuration

### Environment Parameters

Configurable in `irsim_env.py` or passed to `gym.make()`:
- `randomize_start`: Randomize robot starting position (default: True)
- `randomize_goal`: Randomize goal position (default: True)
- `min_goal_distance`: Minimum distance between start and goal (default: 5.0m)
- `safe_margin`: Safety margin for spawn locations - keeps robot/goal away from obstacles and walls (default: 0.5m)
- `goal_tolerance`: Distance threshold for goal achievement (default: 0.5m)

### DQN Hyperparameters

Key training parameters in `train_dqn_irsim.py`:
- Learning rate: 0.0001
- Discount factor (γ): 0.99
- Epsilon decay: 0.995 (1.0 → 0.05)
- Batch size: 128
- Replay buffer capacity: 50,000
- Target network update frequency: Every 10 episodes
- Max episode steps: 300

## Network Architecture

The DQN uses a simple fully connected architecture with a separate target network:

```
Input (28) → FC(256) → ReLU → FC(256) → ReLU → FC(128) → ReLU → Output(7)
```

Key features:
- 4 fully connected layers
- ReLU activation functions
- Gradient clipping (max norm: 1.0)
- Smooth L1 Loss (Huber loss)
- Adam optimizer
- Target network updated every 10 episodes

## Results

After training, the agent demonstrates:
- Somewhat efficient navigation from start to goal positions
- Obstacle avoidance using LiDAR perception
- Adaptability to randomized environments
- Typical success rate of 70-90% after 1000 episodes

Training metrics are automatically saved to `checkpoint/training_results.png` for analysis.

**Training Time:** Approximately 15 minutes for 1000 episodes on NVIDIA RTX 5070 Ti, AMD Ryzen 9 9800X3D, 64GB RAM, Samsung 990 Pro SSD.

## Custom Environments

You can create custom maze environments by modifying YAML files in the `world/` directory. See `irsim_world.yaml` for the configuration structure:
- Robot initial state and parameters
- Goal position
- Obstacles (position, size, shape)
- World dimensions
- Sensor configuration (LiDAR settings)

## Troubleshooting

**Velocity clipping warnings:**
The robot is attempting turns that exceed the velocity limits. Reduce `max_angular` in `create_action_map()` to 1.0 or lower.

**CUDA out of memory:**
Reduce the batch size in the `DQNAgent` class, or train on CPU by not specifying GPU.

**Poor training performance:**
Try adjusting:
- Number of training episodes (increase for more learning)
- Reward function parameters
- Epsilon decay rate
- Network architecture

## Acknowledgments

- Built on the [IR-SIM](https://github.com/hanruihua/ir-sim) robot simulator
- Uses the [Gymnasium](https://gymnasium.farama.org/) framework
- Implements the DQN algorithm with target network and experience replay
- Inspired by ["Federated Reinforcement Learning Framework for Mobile Robot Navigation Using ROS and Gazebo"](https://ieeexplore.ieee.org/document/11025184) (An et al., IEEE Internet of Things Magazine, Volume: 8, Issue: 5, September 2025)

## License

MIT License - see LICENSE file for details.

This project uses IR-SIM robot simulator. Check IR-SIM's license for any restrictions on simulator usage.

---

**Note**: This project is intended for learning and experimentation purposes.
