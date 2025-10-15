import os
import sys

# Set environment before imports
os.environ['LOGURU_LEVEL'] = 'ERROR'

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import copy
import multiprocessing
from multiprocessing import Process, Queue, Manager
import gymnasium as gym
from tqdm import tqdm
import warnings

# Force spawn method to avoid fork issues with graphics
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

warnings.filterwarnings('ignore')

# Suppress loguru
try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
except ImportError:
    pass

import irsim
import irsim_env

# Import from existing training script
from train_dqn_irsim import (
    DQN, 
    DQNAgent, 
    ReplayBuffer,
    create_action_map, 
    flatten_obs, 
    calculate_reward
)


class FederatedServer:
    """Central server for federated learning aggregation."""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.global_model = DQN(state_size, action_size)
        self.round = 0
        
    def aggregate_weights(self, client_weights, client_metrics=None):
        """
        Aggregate weights from multiple clients using FedAvg algorithm.
        
        Args:
            client_weights: List of state_dicts from clients (on CPU)
            client_metrics: Optional list of metrics for weighted averaging
        """
        if not client_weights:
            return
        
        # Ensure all weights are on CPU
        client_weights = [
            {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in w.items()}
            for w in client_weights
        ]
        
        # Simple averaging (FedAvg)
        if client_metrics is None:
            avg_weights = {}
            for key in client_weights[0].keys():
                avg_weights[key] = torch.stack([w[key].float() for w in client_weights]).mean(0)
        else:
            # Weighted averaging based on client performance
            total_weight = sum(client_metrics)
            if total_weight == 0:
                total_weight = 1.0
            avg_weights = {}
            for key in client_weights[0].keys():
                weighted_sum = sum(
                    w[key].float() * (metric / total_weight) 
                    for w, metric in zip(client_weights, client_metrics)
                )
                avg_weights[key] = weighted_sum
        
        self.global_model.load_state_dict(avg_weights)
        self.round += 1
        return avg_weights
    
    def get_global_model(self):
        """Return current global model state (CPU for safe transfer)."""
        cpu_weights = {}
        for key, value in self.global_model.state_dict().items():
            cpu_weights[key] = value.cpu().clone()
        return cpu_weights


class FederatedDQNAgent(DQNAgent):
    """Extended DQN Agent for federated learning."""
    
    def __init__(self, client_id, state_size, action_size, action_map):
        super().__init__(state_size, action_size, action_map)
        self.client_id = client_id
        self.local_episodes = 0
        self.performance_metric = 0.0  # For weighted aggregation
    
    def get_weights(self):
        """Get local model weights (CPU for safe transfer)."""
        cpu_weights = {}
        for key, value in self.model.state_dict().items():
            cpu_weights[key] = value.cpu().clone()
        return cpu_weights
    
    def set_weights(self, weights):
        """Update local model with global weights."""
        # Move weights to device
        device_weights = {}
        for key, value in weights.items():
            device_weights[key] = value.to(self.device)
        self.model.load_state_dict(device_weights)
        self.target_model.load_state_dict(device_weights)
    
    def update_performance_metric(self, success_rate):
        """Update performance metric for weighted averaging."""
        self.performance_metric = success_rate


def train_local_client(client_id, world_config, episodes_per_round, 
                       global_weights_queue, local_weights_queue, 
                       metrics_queue, shared_stats):
    """
    Train a local client (robot) for specified episodes.
    
    Args:
        client_id: Unique identifier for this client
        world_config: Path to world configuration YAML
        episodes_per_round: Number of episodes between synchronization
        global_weights_queue: Queue to receive global weights
        local_weights_queue: Queue to send local weights
        metrics_queue: Queue to send performance metrics
        shared_stats: Shared dictionary for logging
    """
    # Import inside function for spawn multiprocessing
    import os
    import warnings
    import gymnasium as gym
    import numpy as np
    
    os.environ['LOGURU_LEVEL'] = 'ERROR'
    warnings.filterwarnings('ignore')
    
    try:
        from loguru import logger
        logger.remove()
    except:
        pass
    
    import irsim
    import irsim_env
    from train_dqn_irsim import create_action_map, flatten_obs, calculate_reward
    
    try:
        # Setup environment
        world_path = os.path.join('world', world_config)
        env = gym.make('gymnasium_env/IRSIM-v0',
                      world_config=world_path,
                      render_mode=None,
                      randomize_start=True,
                      randomize_goal=True,
                      min_goal_distance=5.0,
                      safe_margin=0.3,
                      goal_tolerance=0.5)
        
        # Setup agent
        obs, _ = env.reset()
        action_map = create_action_map(env.unwrapped.max_angular, 0.5)
        state_size = len(flatten_obs(obs))
        action_size = len(action_map)
        
        agent = FederatedDQNAgent(client_id, state_size, action_size, action_map)
        
        round_num = 0
        while True:
            # Check for global weights update
            if not global_weights_queue.empty():
                global_weights = global_weights_queue.get()
                if global_weights is None:  # Termination signal
                    break
                agent.set_weights(global_weights)
                round_num += 1
            
            # Local training
            local_scores = []
            local_successes = 0
            
            for ep in range(episodes_per_round):
                obs, info = env.reset()
                state = flatten_obs(obs)
                total_reward = 0
                episode_success = False
                
                for step in range(300):
                    discrete_action = agent.act(state)
                    continuous_action = action_map[discrete_action]
                    
                    prev_obs, prev_info = obs, info
                    next_obs, _, terminated, truncated, info = env.step(continuous_action)
                    next_state = flatten_obs(next_obs)
                    
                    reward = calculate_reward(prev_obs, next_obs, prev_info, info)
                    done = terminated or truncated or info['collision'] or info['goal_reached']
                    
                    agent.memory.push(state, discrete_action, reward, next_state, done)
                    agent.train()
                    
                    state = next_state
                    obs = next_obs
                    total_reward += reward
                    
                    if info['goal_reached'] and not episode_success:
                        local_successes += 1
                        episode_success = True
                    
                    if done:
                        break
                
                local_scores.append(total_reward)
                agent.local_episodes += 1
                
                # Update target network periodically
                if agent.local_episodes % agent.target_update == 0:
                    agent.update_target_network()
            
            # Send local weights and metrics
            success_rate = local_successes / episodes_per_round
            agent.update_performance_metric(success_rate)
            
            # Get weights on CPU for safe transfer
            local_weights = agent.get_weights()
            local_weights_queue.put((client_id, local_weights))
            metrics_queue.put((client_id, success_rate, np.mean(local_scores)))
            
            # Update shared stats (optional, may fail if connection closed)
            try:
                shared_stats[f'client_{client_id}_success'] = success_rate
                shared_stats[f'client_{client_id}_reward'] = np.mean(local_scores)
            except:
                pass  # Ignore if connection closed
        
        env.close()
        
    except Exception as e:
        print(f"Client {client_id} error: {e}")
        import traceback
        traceback.print_exc()


def train_federated_dqn(num_clients=4, rounds=50, episodes_per_round=20,
                       world_configs=None, aggregation='simple'):
    """
    Train DQN using federated learning.
    
    Args:
        num_clients: Number of robots/clients
        rounds: Number of federated learning rounds
        episodes_per_round: Episodes each client trains before synchronization
        world_configs: List of world configs (one per client), or None for same world
        aggregation: 'simple' for FedAvg, 'weighted' for performance-based
    """
    
    print(f"\n{'='*70}")
    print(f"Federated DQN Training")
    print(f"{'='*70}")
    print(f"Clients: {num_clients}")
    print(f"Rounds: {rounds}")
    print(f"Episodes per round: {episodes_per_round}")
    print(f"Total episodes per client: {rounds * episodes_per_round}")
    print(f"Aggregation: {aggregation}")
    print(f"World configs: {world_configs}")
    print(f"{'='*70}\n")
    
    # Setup world configurations
    if world_configs is None:
        world_configs = ['irsim_world.yaml'] * num_clients
    elif len(world_configs) < num_clients:
        # Repeat configs to match number of clients
        world_configs = (world_configs * (num_clients // len(world_configs) + 1))[:num_clients]
    
    print(f"Assigned worlds: {world_configs}\n")
    
    # Initialize server
    # Get state/action size from a sample environment
    sample_env = gym.make('gymnasium_env/IRSIM-v0',
                         world_config=os.path.join('world', world_configs[0]))
    obs, _ = sample_env.reset()
    state_size = len(flatten_obs(obs))
    action_size = 7  # From create_action_map
    sample_env.close()
    
    print(f"State size: {state_size}, Action size: {action_size}\n")
    
    server = FederatedServer(state_size, action_size)
    
    # Setup multiprocessing
    manager = Manager()
    shared_stats = manager.dict()
    
    global_weights_queues = [Queue() for _ in range(num_clients)]
    local_weights_queue = Queue()
    metrics_queue = Queue()
    
    # Start client processes
    processes = []
    print("Starting client processes...")
    for i in range(num_clients):
        p = Process(target=train_local_client,
                   args=(i, world_configs[i], episodes_per_round,
                        global_weights_queues[i], local_weights_queue,
                        metrics_queue, shared_stats))
        p.start()
        processes.append(p)
        print(f"  Client {i} started (world: {world_configs[i]})")
    
    print(f"\nStarting federated training...\n")
    
    # Federated learning rounds
    all_round_metrics = []
    
    pbar = tqdm(range(rounds), desc="FL Rounds", unit="round")
    
    for round_num in pbar:
        # Broadcast global model to all clients
        global_weights = server.get_global_model()
        for q in global_weights_queues:
            q.put(global_weights)
        
        # Collect local weights and metrics from all clients
        client_weights = []
        client_metrics = []
        round_stats = {}
        
        for _ in range(num_clients):
            client_id, weights = local_weights_queue.get()
            client_weights.append(weights)
            
            cid, success, reward = metrics_queue.get()
            client_metrics.append(max(success, 0.01) if aggregation == 'weighted' else 1.0)
            round_stats[f'client_{cid}'] = {'success': success, 'reward': reward}
        
        # Aggregate weights
        if aggregation == 'weighted':
            server.aggregate_weights(client_weights, client_metrics)
        else:
            server.aggregate_weights(client_weights)
        
        # Log statistics
        avg_success = np.mean([s['success'] for s in round_stats.values()])
        avg_reward = np.mean([s['reward'] for s in round_stats.values()])
        all_round_metrics.append({
            'round': round_num,
            'avg_success': avg_success,
            'avg_reward': avg_reward,
            'clients': round_stats
        })
        
        pbar.set_postfix({
            'success': f'{avg_success:.2%}',
            'reward': f'{avg_reward:.1f}'
        })
        
        # Save checkpoint
        if round_num % 10 == 0 and round_num > 0:
            # Save with CPU tensors
            cpu_state_dict = {k: v.cpu() for k, v in server.global_model.state_dict().items()}
            torch.save({
                'round': round_num,
                'model_state_dict': cpu_state_dict,
                'metrics': all_round_metrics,
                'state_size': state_size,
                'action_size': action_size
            }, f'checkpoint/federated_round_{round_num}.pth')
    
    pbar.close()
    
    # Terminate clients
    print("\nTerminating client processes...")
    for q in global_weights_queues:
        q.put(None)
    
    for i, p in enumerate(processes):
        p.join(timeout=5)
        if p.is_alive():
            print(f"  Force terminating client {i}")
            p.terminate()
        else:
            print(f"  Client {i} terminated gracefully")
    
    # Save final model
    cpu_state_dict = {k: v.cpu() for k, v in server.global_model.state_dict().items()}
    torch.save({
        'model_state_dict': cpu_state_dict,
        'metrics': all_round_metrics,
        'num_clients': num_clients,
        'rounds': rounds,
        'state_size': state_size,
        'action_size': action_size
    }, 'checkpoint/federated_final.pth')
    
    print(f"\n{'='*70}")
    print(f"Federated Training Complete")
    print(f"{'='*70}")
    print(f"Final average success rate: {all_round_metrics[-1]['avg_success']:.2%}")
    print(f"Final average reward: {all_round_metrics[-1]['avg_reward']:.2f}")
    print(f"\nPer-client final performance:")
    for client_name, stats in all_round_metrics[-1]['clients'].items():
        print(f"  {client_name}: Success={stats['success']:.2%}, Reward={stats['reward']:.2f}")
    print(f"\nModel saved to 'checkpoint/federated_final.pth'")
    print(f"{'='*70}\n")
    
    return server, all_round_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Federated DQN Training')
    parser.add_argument('--clients', type=int, default=4, help='Number of clients')
    parser.add_argument('--rounds', type=int, default=50, help='Federated rounds')
    parser.add_argument('--episodes', type=int, default=20, help='Episodes per round')
    parser.add_argument('--worlds', nargs='+', default=None,
                       help='World configs (e.g., world1.yaml world2.yaml)')
    parser.add_argument('--aggregation', choices=['simple', 'weighted'], 
                       default='simple', help='Aggregation method')
    
    args = parser.parse_args()
    
    server, metrics = train_federated_dqn(
        num_clients=args.clients,
        rounds=args.rounds,
        episodes_per_round=args.episodes,
        world_configs=args.worlds,
        aggregation=args.aggregation
    )