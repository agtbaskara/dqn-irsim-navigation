import os
import sys
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import argparse
from tqdm import tqdm
import warnings

# Suppress logging
os.environ['LOGURU_LEVEL'] = 'CRITICAL'
warnings.filterwarnings('ignore')

try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="CRITICAL",
               filter=lambda record: "velocity" not in record["message"].lower() 
               and "collided" not in record["message"].lower())
except ImportError:
    pass

import irsim
import irsim_env


# ============== DQN Network ==============

class DQN(nn.Module):
    """Deep Q-Network with 4 fully connected layers."""
    
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# ============== Agent ==============

class DQNAgent:
    """DQN Agent for testing."""
    
    def __init__(self, state_size, action_size, action_map):
        self.state_size = state_size
        self.action_size = action_size
        self.action_map = action_map
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.epsilon = 0.0  # Greedy policy
    
    def act(self, state):
        """Select action using greedy policy."""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            return q_values.argmax().item()
    
    def load(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'episode' in checkpoint:
                print(f"Loaded checkpoint from episode {checkpoint['episode']}")
        else:
            self.model.load_state_dict(checkpoint)
            print("Loaded model state dict")
        
        self.model.eval()


# ============== Helper Functions ==============

def create_action_map(max_angular=2.0, constant_linear=0.5):
    """Create discrete action space with limits to avoid clipping."""
    max_angular = min(max_angular, 1.0)  # Limit to avoid acceleration warnings
    
    angular_velocities = [
        -max_angular, -max_angular * 0.66, -max_angular * 0.33,
        0.0,
        max_angular * 0.33, max_angular * 0.66, max_angular
    ]
    actions = [[constant_linear, ang_vel] for ang_vel in angular_velocities]
    return {i: np.array(action, dtype=np.float32) for i, action in enumerate(actions)}


def flatten_obs(obs):
    """Convert observation to flat state vector."""
    robot = obs['robot_location']
    goal = obs['goal_location']
    lidar = obs['lidar']
    
    # Ensure 24 LiDAR beams
    if len(lidar) != 24:
        indices = np.linspace(0, len(lidar) - 1, 24, dtype=int)
        lidar = lidar[indices]
    
    lidar_norm = np.clip(lidar / 10.0, 0, 1)
    
    # Goal features
    dx, dy = goal[0] - robot[0], goal[1] - robot[1]
    dist_goal = np.sqrt(dx**2 + dy**2) / 15.0
    angle_goal = np.arctan2(dy, dx) - robot[2]
    angle_goal = np.arctan2(np.sin(angle_goal), np.cos(angle_goal)) / np.pi
    
    # Obstacle features
    min_idx = np.argmin(lidar)
    dist_obstacle = lidar[min_idx] / 10.0
    angle_obstacle = (min_idx * 2 * np.pi / 24)
    if angle_obstacle > np.pi:
        angle_obstacle -= 2 * np.pi
    angle_obstacle /= np.pi
    
    return np.concatenate([
        lidar_norm, [dist_goal, angle_goal, dist_obstacle, angle_obstacle]
    ]).astype(np.float32)


# ============== Test Function ==============

def test_agent(checkpoint_path, episodes=10, render=True, randomize=True,
               world_config='irsim_world.yaml', seed=None):
    """Test agent from checkpoint."""
    
    # Fixed parameters
    goal_tolerance = 0.5
    constant_linear = 0.5
    
    # Load from checkpoint/ and world/ folders
    checkpoint_path = os.path.join('checkpoint', checkpoint_path)
    world_path = os.path.join('world', world_config)
    
    print(f"\n{'='*70}")
    print(f"Testing DQN Agent")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"World config: {world_path}")
    print(f"Episodes: {episodes}")
    print(f"Randomize: {randomize}")
    print(f"Seed: {seed if seed is not None else 'None (random)'}")
    print(f"{'='*70}\n")
    
    # Create environment
    env = gym.make('gymnasium_env/IRSIM-v0',
                   world_config=world_path,
                   render_mode='human' if render else None,
                   randomize_start=randomize,
                   randomize_goal=randomize,
                   min_goal_distance=5.0,
                   safe_margin=0.3,
                   goal_tolerance=goal_tolerance)
    
    # Setup agent
    obs, _ = env.reset(seed=seed)
    action_map = create_action_map(env.unwrapped.max_angular, constant_linear)
    state_size = len(flatten_obs(obs))
    action_size = len(action_map)
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"LiDAR beams: {len(obs['lidar'])}\n")
    
    agent = DQNAgent(state_size, action_size, action_map)
    agent.load(checkpoint_path)
    
    # Test
    results = {
        'success': 0,
        'collision': 0,
        'timeout': 0,
        'steps': [],
        'rewards': []
    }
    
    pbar = tqdm(range(episodes), desc="Testing", unit="ep")
    
    for episode in pbar:
        # Use seed + episode number for consistent but different episodes
        episode_seed = seed + episode if seed is not None else None
        obs, info = env.reset(seed=episode_seed)
        state = flatten_obs(obs)
        total_reward = 0
        steps = 0
        
        for step in range(300):
            discrete_action = agent.act(state)
            continuous_action = action_map[discrete_action]
            
            obs, reward, terminated, truncated, info = env.step(continuous_action)
            state = flatten_obs(obs)
            
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
            
            # Check termination
            if terminated or truncated or info['collision'] or info['goal_reached']:
                if info['goal_reached']:
                    results['success'] += 1
                elif info['collision']:
                    results['collision'] += 1
                else:
                    results['timeout'] += 1
                break
        else:
            results['timeout'] += 1
        
        results['steps'].append(steps)
        results['rewards'].append(total_reward)
        
        pbar.set_postfix({
            'success': f"{results['success']}/{episode+1}",
            'avg_steps': f"{np.mean(results['steps']):.1f}"
        })
    
    pbar.close()
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Test Results")
    print(f"{'='*70}")
    print(f"Episodes: {episodes}")
    print(f"Success: {results['success']} ({results['success']/episodes*100:.1f}%)")
    print(f"Collision: {results['collision']} ({results['collision']/episodes*100:.1f}%)")
    print(f"Timeout: {results['timeout']} ({results['timeout']/episodes*100:.1f}%)")
    print(f"Avg Steps: {np.mean(results['steps']):.1f}")
    print(f"Avg Reward: {np.mean(results['rewards']):.2f}")
    print(f"{'='*70}\n")
    
    env.close()
    return results


# ============== Main ==============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test DQN agent from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='dqn_final.pth',
                       help='Checkpoint file in checkpoint/ folder')
    parser.add_argument('--episodes', type=int, default=10, help='Test episodes')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--no-randomize', action='store_true', help='Fixed positions')
    parser.add_argument('--world', type=str, default='irsim_world.yaml',
                       help='World config file in world/ folder')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Run test
    test_agent(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        render=not args.no_render,
        randomize=not args.no_randomize,
        world_config=args.world,
        seed=args.seed
    )