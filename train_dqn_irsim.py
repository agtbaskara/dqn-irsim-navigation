import os
import sys

# Set environment variable to suppress loguru before ANY imports
os.environ['LOGURU_LEVEL'] = 'ERROR'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

# Suppress all warnings
warnings.filterwarnings('ignore')

# Configure loguru aggressively
try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    # Try to disable loguru completely for lower levels
    logger.disable("irsim")
    logger.disable("")  # Disable all modules
except ImportError:
    pass

# Now import IRSIM
import irsim
import irsim_env

# Final attempt: patch loguru after IRSIM import
try:
    from loguru import logger as loguru_logger
    # Remove all handlers and add only ERROR level
    loguru_logger.remove()
    loguru_logger.add(sys.stderr, level="ERROR")
except:
    pass

# Create checkpoint directory
os.makedirs('checkpoint', exist_ok=True)


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


# ============== Replay Buffer ==============

class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


# ============== DQN Agent ==============

class DQNAgent:
    """DQN Agent with target network."""
    
    def __init__(self, state_size, action_size, action_map):
        self.state_size = state_size
        self.action_size = action_size
        self.action_map = action_map
        self.memory = ReplayBuffer()
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.batch_size = 128
        self.target_update = 10
        
        # Networks
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()
    
    def act(self, state):
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            return q_values.argmax().item()
    
    def train(self):
        """Train the network on a batch from replay buffer."""
        if len(self.memory) < self.batch_size:
            return None
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values using vanilla DQN
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
        
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_model.load_state_dict(self.model.state_dict())


# ============== Helper Functions ==============

def create_action_map(max_angular=2.0, constant_linear=0.5):
    """
    Create discrete action space with constant linear velocity.
    Returns 7 actions: straight + 3 levels of left/right turns.
    
    Note: If you see velocity clipping warnings, reduce max_angular to 1.0
    """
    # Clamp to reasonable limits to avoid acceleration warnings
    max_angular = min(max_angular, 1.0)  # Limit to avoid clipping warnings
    
    angular_velocities = [
        -max_angular,          # Strong right
        -max_angular * 0.66,   # Medium right
        -max_angular * 0.33,   # Weak right
        0.0,                   # Straight
        max_angular * 0.33,    # Weak left
        max_angular * 0.66,    # Medium left
        max_angular,           # Strong left
    ]
    
    actions = [[constant_linear, ang_vel] for ang_vel in angular_velocities]
    return {i: np.array(action, dtype=np.float32) for i, action in enumerate(actions)}


def flatten_obs(obs):
    """
    Convert observation to flat state vector (28 features):
    - 24 LiDAR readings (normalized)
    - Distance to goal (normalized)
    - Angle to goal (normalized)
    - Distance to nearest obstacle (normalized)
    - Angle to nearest obstacle (normalized)
    """
    robot = obs['robot_location']
    goal = obs['goal_location']
    lidar = obs['lidar']
    
    # Ensure 24 LiDAR beams
    if len(lidar) != 24:
        indices = np.linspace(0, len(lidar) - 1, 24, dtype=int)
        lidar = lidar[indices]
    
    # Normalize LiDAR (max range 10m)
    lidar_norm = np.clip(lidar / 10.0, 0, 1)
    
    # Distance and angle to goal
    dx, dy = goal[0] - robot[0], goal[1] - robot[1]
    dist_goal = np.sqrt(dx**2 + dy**2) / 15.0  # Normalize
    angle_goal = np.arctan2(dy, dx) - robot[2]
    angle_goal = np.arctan2(np.sin(angle_goal), np.cos(angle_goal)) / np.pi
    
    # Nearest obstacle
    min_idx = np.argmin(lidar)
    dist_obstacle = lidar[min_idx] / 10.0
    angle_obstacle = (min_idx * 2 * np.pi / 24)
    if angle_obstacle > np.pi:
        angle_obstacle -= 2 * np.pi
    angle_obstacle /= np.pi
    
    return np.concatenate([
        lidar_norm,
        [dist_goal, angle_goal, dist_obstacle, angle_obstacle]
    ]).astype(np.float32)


def calculate_reward(obs, next_obs, info, next_info):
    """
    Calculate reward: R = R_θ × R_d + R_obstacle
    - R_θ: Angle reward (facing goal)
    - R_d: Distance reward (progress toward goal)
    - R_obstacle: Penalty for being close to obstacles
    """
    # Terminal rewards
    if next_info['goal_reached']:
        return 100.0
    if next_info['collision']:
        return -100.0
    
    robot = next_obs['robot_location']
    goal = next_obs['goal_location']
    
    # Angle reward
    dx, dy = goal[0] - robot[0], goal[1] - robot[1]
    angle_to_goal = np.arctan2(dy, dx) - robot[2]
    angle_to_goal = np.arctan2(np.sin(angle_to_goal), np.cos(angle_to_goal))
    R_theta = np.cos(angle_to_goal) + 1.0  # [0, 2]
    
    # Distance reward
    distance_change = info['distance_to_goal'] - next_info['distance_to_goal']
    R_distance = distance_change * 10.0
    
    # Obstacle penalty
    min_dist = np.min(next_obs['lidar'])
    if min_dist < 0.3:
        R_obstacle = -5.0 / (min_dist + 0.1)
    elif min_dist < 0.5:
        R_obstacle = -2.0 / (min_dist + 0.1)
    else:
        R_obstacle = 0.0
    
    return R_theta * R_distance + R_obstacle


# ============== Training ==============

def train_dqn(episodes=1000, render=False, randomize=True, 
              world_config='irsim_world.yaml'):
    """Train DQN agent on IRSIM environment."""
    
    # Fixed parameters
    goal_tolerance = 0.5
    constant_linear = 0.5
    
    # Load world config from world/ folder
    world_path = os.path.join('world', world_config)
    
    # Create environment
    env = gym.make('gymnasium_env/IRSIM-v0',
                   world_config=world_path,
                   render_mode='human' if render else None,
                   randomize_start=randomize,
                   randomize_goal=randomize,
                   min_goal_distance=5.0,
                   safe_margin=0.3,
                   goal_tolerance=goal_tolerance)
    
    # Setup
    obs, info = env.reset()
    action_map = create_action_map(env.unwrapped.max_angular, constant_linear)
    state_size = len(flatten_obs(obs))
    action_size = len(action_map)
    
    print(f"\n{'='*70}")
    print(f"DQN Training - IRSIM Environment")
    print(f"{'='*70}")
    print(f"World config: {world_path}")
    print(f"State size: {state_size} (24 LiDAR + 4 features)")
    print(f"Action size: {action_size} discrete actions")
    print(f"Randomization: {'ON' if randomize else 'OFF'}")
    print(f"{'='*70}\n")
    
    agent = DQNAgent(state_size, action_size, action_map)
    
    # Training metrics
    scores = []
    steps_list = []
    losses = []
    success_count = 0
    collision_count = 0
    
    pbar = tqdm(range(episodes), desc="Training", unit="ep")
    
    for episode in pbar:
        obs, info = env.reset()
        state = flatten_obs(obs)
        total_reward = 0
        steps = 0
        episode_losses = []
        episode_success = False
        had_collision = False
        
        for step in range(300):
            # Select and perform action
            discrete_action = agent.act(state)
            continuous_action = action_map[discrete_action]
            
            prev_obs, prev_info = obs, info
            next_obs, _, terminated, truncated, info = env.step(continuous_action)
            next_state = flatten_obs(next_obs)
            
            # Calculate custom reward
            reward = calculate_reward(prev_obs, next_obs, prev_info, info)
            
            # Check termination
            done = terminated or truncated or info['collision'] or info['goal_reached']
            
            # Track metrics
            if info['collision'] and not had_collision:
                collision_count += 1
                had_collision = True
            if info['goal_reached'] and not had_collision:
                success_count += 1
                episode_success = True
            
            # Store and train
            agent.memory.push(state, discrete_action, reward, next_state, done)
            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            obs = next_obs
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
            
            if done:
                break
        
        # Update target network
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        # Record metrics
        scores.append(total_reward)
        steps_list.append(steps)
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
        # Update progress bar
        if episode % 5 == 0:
            avg_score = np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores)
            avg_steps = np.mean(steps_list[-20:]) if len(steps_list) >= 20 else np.mean(steps_list)
            recent_success = sum(1 for i in range(max(0, episode-19), episode+1) 
                               if i < len(scores) and scores[i] > 50)
            
            pbar.set_postfix({
                'reward': f'{avg_score:.1f}',
                'steps': f'{avg_steps:.0f}',
                'success': f'{recent_success}/20',
                'ε': f'{agent.epsilon:.3f}'
            })
        
        # Save checkpoint
        if episode % 100 == 0 and episode > 0:
            torch.save({
                'episode': episode,
                'model_state_dict': agent.model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
            }, f'checkpoint/checkpoint_{episode}.pth')
    
    pbar.close()
    
    print(f"\n{'='*70}")
    print(f"Training Complete")
    print(f"{'='*70}")
    print(f"Successes: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
    print(f"Collisions: {collision_count}/{episodes} ({collision_count/episodes*100:.1f}%)")
    print(f"{'='*70}\n")
    
    env.close()
    return agent, scores, steps_list, losses


# ============== Testing ==============

def test_agent(agent, episodes=10, render=True, randomize=True, 
               world_config='irsim_world.yaml'):
    """Test trained agent."""
    
    # Fixed parameters
    goal_tolerance = 0.5
    
    # Load world config from world/ folder
    world_path = os.path.join('world', world_config)
    
    env = gym.make('gymnasium_env/IRSIM-v0',
                   world_config=world_path,
                   render_mode='human' if render else None,
                   randomize_start=randomize,
                   randomize_goal=randomize,
                   min_goal_distance=5.0,
                   safe_margin=0.3,
                   goal_tolerance=goal_tolerance)
    
    agent.epsilon = 0  # Greedy policy
    
    print(f"\n{'='*70}")
    print(f"Testing Agent")
    print(f"{'='*70}")
    print(f"World config: {world_path}\n")
    
    success_count = 0
    collision_count = 0
    steps_list = []
    
    pbar = tqdm(range(episodes), desc="Testing", unit="ep")
    
    for episode in pbar:
        obs, info = env.reset()
        state = flatten_obs(obs)
        steps = 0
        
        for step in range(300):
            discrete_action = agent.act(state)
            continuous_action = agent.action_map[discrete_action]
            
            obs, _, terminated, truncated, info = env.step(continuous_action)
            state = flatten_obs(obs)
            steps += 1
            
            if render:
                env.render()
            
            if terminated or truncated or info['collision'] or info['goal_reached']:
                if info['goal_reached']:
                    success_count += 1
                elif info['collision']:
                    collision_count += 1
                break
        
        steps_list.append(steps)
        pbar.set_postfix({
            'success': f'{success_count}/{episode+1}',
            'avg_steps': f'{np.mean(steps_list):.1f}'
        })
    
    pbar.close()
    
    print(f"\n{'='*70}")
    print(f"Test Results")
    print(f"{'='*70}")
    print(f"Success: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
    print(f"Collision: {collision_count}/{episodes} ({collision_count/episodes*100:.1f}%)")
    print(f"Avg Steps: {np.mean(steps_list):.1f}")
    print(f"{'='*70}\n")
    
    env.close()


# ============== Plotting ==============

def plot_results(scores, steps_list, losses):
    """Plot training results."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    window = 20
    
    # Rewards
    axes[0].plot(scores, alpha=0.3, color='blue')
    if len(scores) >= window:
        moving_avg = [np.mean(scores[max(0, i-window):i+1]) for i in range(len(scores))]
        axes[0].plot(moving_avg, color='red', linewidth=2, label=f'{window}-ep avg')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Training Rewards')
    axes[0].legend()
    axes[0].grid(True)
    
    # Steps
    axes[1].plot(steps_list, alpha=0.3, color='green')
    if len(steps_list) >= window:
        moving_avg = [np.mean(steps_list[max(0, i-window):i+1]) for i in range(len(steps_list))]
        axes[1].plot(moving_avg, color='red', linewidth=2, label=f'{window}-ep avg')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    axes[1].set_title('Episode Length')
    axes[1].legend()
    axes[1].grid(True)
    
    # Loss
    if losses:
        axes[2].plot(losses, alpha=0.6, color='purple')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Training Loss')
        axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('checkpoint/training_results.png', dpi=150)
    print("Plot saved to 'checkpoint/training_results.png'")


# ============== Main ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN on IRSIM')
    parser.add_argument('--episodes', type=int, default=1000, help='Training episodes')
    parser.add_argument('--render', action='store_true', help='Render during training')
    parser.add_argument('--no-randomize', action='store_true', help='Fixed start/goal')
    parser.add_argument('--world', type=str, default='irsim_world.yaml', 
                       help='World config file in world/ folder')
    
    args = parser.parse_args()
    
    # Train
    agent, scores, steps_list, losses = train_dqn(
        episodes=args.episodes,
        render=args.render,
        randomize=not args.no_randomize,
        world_config=args.world
    )
    
    # Save model
    torch.save({
        'model_state_dict': agent.model.state_dict(),
        'action_map': agent.action_map,
        'state_size': agent.state_size,
        'action_size': agent.action_size,
    }, 'checkpoint/dqn_final.pth')
    print("Model saved to 'checkpoint/dqn_final.pth'")
    
    # Plot
    plot_results(scores, steps_list, losses)
    
    # Test
    test_agent(agent, episodes=5, render=args.render, 
               randomize=not args.no_randomize,
               world_config=args.world)