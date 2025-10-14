from typing import Optional
import numpy as np
import gymnasium as gym
import irsim
import yaml
import custom_behavior


class IRSIMEnv(gym.Env):
    """Gymnasium wrapper for IR-SIM robot simulator with randomization support."""
    
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode: Optional[str] = None, world_config: str = 'irsim_world.yaml',
                 randomize_start: bool = True, randomize_goal: bool = True, 
                 min_goal_distance: float = 5.0, safe_margin: float = 0.5,
                 goal_tolerance: float = 0.5):
        
        self.render_mode = render_mode
        self.world_config = world_config
        self.randomize_start = randomize_start
        self.randomize_goal = randomize_goal
        self.min_goal_distance = min_goal_distance
        self.safe_margin = safe_margin
        self.goal_tolerance = goal_tolerance
        
        # Load IRSIM environment
        print(f"Loading world: {world_config}")
        self.irsim_env = irsim.make(world_config)
        self.robot = self.irsim_env.robot
        self.world = self.irsim_env._world if hasattr(self.irsim_env, '_world') else None
        
        # Store default positions from YAML
        self.default_robot_state = self.robot.state.flatten()[:3].copy()
        self.default_goal_state = self.robot.goal.flatten()[:3].copy() if hasattr(self.robot, 'goal') else np.array([9.0, 9.0, 0.0])
        
        # Get robot parameters
        self.robot_radius = self._get_robot_radius()
        self.world_width = float(self.irsim_env._world.width)
        self.world_height = float(self.irsim_env._world.height)
        
        # Load obstacles and configure sensors
        self.obstacles = self._load_obstacles_from_yaml(world_config)
        self.lidar, self.lidar_range, self.lidar_num_beams = self._configure_lidar()
        self.max_linear, self.max_angular = self._get_velocity_limits()
        
        # Define spaces
        self.observation_space = self._create_observation_space()
        self.action_space = gym.spaces.Box(
            low=np.array([-self.max_linear, -self.max_angular], dtype=np.float32),
            high=np.array([self.max_linear, self.max_angular], dtype=np.float32),
            dtype=np.float32
        )
    
    def _get_robot_radius(self) -> float:
        """Extract robot radius from configuration."""
        if hasattr(self.robot, 'shape'):
            if hasattr(self.robot.shape, 'radius'):
                return float(self.robot.shape.radius)
            elif isinstance(self.robot.shape, dict) and 'radius' in self.robot.shape:
                return float(self.robot.shape['radius'])
        return 0.2
    
    def _configure_lidar(self) -> tuple:
        """Configure LiDAR sensor parameters."""
        if hasattr(self.robot, 'sensors') and len(self.robot.sensors) > 0:
            lidar = self.robot.sensors[0]
            lidar_range = float(lidar.range_max)
            lidar_num_beams = int(lidar.number)
            return lidar, lidar_range, lidar_num_beams
        return None, 10.0, 100
    
    def _get_velocity_limits(self) -> tuple:
        """Get maximum linear and angular velocities."""
        if hasattr(self.robot, 'vel_max'):
            vel_max = self.robot.vel_max.flatten()
            return float(vel_max[0]), float(vel_max[1])
        return 2.0, 2.0
    
    def _create_observation_space(self) -> gym.spaces.Dict:
        """Create the observation space."""
        return gym.spaces.Dict({
            "robot_location": gym.spaces.Box(
                low=np.array([0.0, 0.0, -np.pi], dtype=np.float32), 
                high=np.array([self.world_width, self.world_height, np.pi], dtype=np.float32),
                dtype=np.float32
            ),
            "goal_location": gym.spaces.Box(
                low=np.array([0.0, 0.0, -np.pi], dtype=np.float32),
                high=np.array([self.world_width, self.world_height, np.pi], dtype=np.float32),
                dtype=np.float32
            ),
            "lidar": gym.spaces.Box(
                low=0.0, high=self.lidar_range,
                shape=(self.lidar_num_beams,), dtype=np.float32
            ),
        })
    
    def _load_obstacles_from_yaml(self, yaml_file: str) -> list:
        """Load obstacle positions and sizes from YAML configuration."""
        obstacles = []
        try:
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
            
            if 'obstacle' in config and config['obstacle']:
                for obs in config['obstacle']:
                    if 'state' not in obs:
                        continue
                    
                    pos = np.array([float(obs['state'][0]), float(obs['state'][1])])
                    size = 0.5
                    
                    if 'shape' in obs:
                        shape = obs['shape']
                        if 'radius' in shape:
                            size = float(shape['radius'])
                        elif 'length' in shape and 'width' in shape:
                            size = min(float(shape['length']), float(shape['width'])) / 2
                        elif 'length' in shape:
                            size = float(shape['length']) / 2
                        elif 'width' in shape:
                            size = float(shape['width']) / 2
                    
                    obstacles.append({'pos': pos, 'size': size})
        except Exception as e:
            print(f"Warning: Could not load obstacles from YAML: {e}")
        
        return obstacles
    
    def _is_valid_robot_position(self, pos: np.ndarray) -> bool:
        """Check if robot position is valid (not in collision with obstacles or walls)."""
        x, y = pos[0], pos[1]
        margin = self.robot_radius + self.safe_margin
        
        # Check world bounds
        if x < margin or x > self.world_width - margin:
            return False
        if y < margin or y > self.world_height - margin:
            return False
        
        # Check obstacles
        for obs in self.obstacles:
            dist = np.linalg.norm(pos[:2] - obs['pos'])
            if dist < self.robot_radius + obs['size'] + self.safe_margin:
                return False
        
        return True
    
    def _is_valid_goal_position(self, pos: np.ndarray) -> bool:
        """Check if goal position is valid (not inside or too near obstacles)."""
        x, y = pos[0], pos[1]
        
        # Use smaller margin for goals - just keep them away from walls
        wall_margin = max(0.3, self.safe_margin)
        
        # Check world bounds
        if x < wall_margin or x > self.world_width - wall_margin:
            return False
        if y < wall_margin or y > self.world_height - wall_margin:
            return False
        
        # Check obstacles - goal must be at least obstacle_size + safe_margin away
        for obs in self.obstacles:
            dist = np.linalg.norm(pos[:2] - obs['pos'])
            # Goal clearance: obstacle radius + safety margin (not robot radius)
            if dist < obs['size'] + self.safe_margin * 2:
                return False
        
        return True
    
    def _sample_valid_robot_position(self, max_attempts: int = 100) -> np.ndarray:
        """Sample a random valid position for the robot."""
        margin = self.robot_radius + self.safe_margin
        
        for _ in range(max_attempts):
            x = np.random.uniform(margin, self.world_width - margin)
            y = np.random.uniform(margin, self.world_height - margin)
            theta = np.random.uniform(-np.pi, np.pi)
            pos = np.array([x, y, theta])
            
            if self._is_valid_robot_position(pos):
                return pos
        
        # Fallback to default
        print("Warning: Could not find valid robot position, using default")
        return self.default_robot_state.copy()
    
    def _sample_valid_goal_position(self, robot_pos: np.ndarray, max_attempts: int = 100) -> np.ndarray:
        """Sample a random valid goal position, ensuring it's far enough from robot."""
        wall_margin = max(0.5, self.safe_margin)
        
        for _ in range(max_attempts):
            x = np.random.uniform(wall_margin, self.world_width - wall_margin)
            y = np.random.uniform(wall_margin, self.world_height - wall_margin)
            theta = np.random.uniform(-np.pi, np.pi)
            pos = np.array([x, y, theta])
            
            # Check if goal position is valid
            if not self._is_valid_goal_position(pos):
                continue
            
            # Check if far enough from robot
            if np.linalg.norm(pos[:2] - robot_pos[:2]) >= self.min_goal_distance:
                return pos
        
        # Fallback to default
        print("Warning: Could not find valid goal position with min distance constraint, using default")
        return self.default_goal_state.copy()
    
    def _get_obs(self) -> dict:
        """Get current observation."""
        robot_location = self.robot.state.flatten()[:3].astype(np.float32)
        goal_location = self.robot.goal.flatten()[:3].astype(np.float32) if hasattr(self.robot, 'goal') else np.zeros(3, dtype=np.float32)
        
        # Get LiDAR readings
        if self.lidar is not None:
            scan_data = self.lidar.get_scan()
            if scan_data and 'ranges' in scan_data:
                lidar_readings = np.array(scan_data['ranges'], dtype=np.float32).flatten()
            else:
                lidar_readings = np.array(self.lidar.range_data, dtype=np.float32).flatten()
        else:
            lidar_readings = np.full(self.lidar_num_beams, self.lidar_range, dtype=np.float32)
        
        # Ensure correct size
        if len(lidar_readings) != self.lidar_num_beams:
            lidar_readings = np.resize(lidar_readings, self.lidar_num_beams).astype(np.float32)
        
        return {
            "robot_location": robot_location,
            "goal_location": goal_location,
            "lidar": lidar_readings
        }
    
    def _get_info(self) -> dict:
        """Compute auxiliary information."""
        robot_pos = self.robot.state.flatten()[:2]
        goal_pos = self.robot.goal.flatten()[:2] if hasattr(self.robot, 'goal') else np.zeros(2)
        distance_to_goal = np.linalg.norm(robot_pos - goal_pos)
        
        # Check collision and goal status
        collision = bool(self.robot.collision_flag) if hasattr(self.robot, 'collision_flag') else False
        goal_reached = (distance_to_goal <= self.goal_tolerance) or \
                      (bool(self.robot.arrive_flag) if hasattr(self.robot, 'arrive_flag') else False)
        
        return {
            "distance_to_goal": float(distance_to_goal),
            "collision": collision,
            "goal_reached": goal_reached,
        }
    
    def _set_robot_state(self, state: np.ndarray):
        """Set robot state."""
        try:
            current_state = self.robot.state
            new_state = state.reshape(current_state.shape).astype(current_state.dtype)
            np.copyto(current_state, new_state)
        except Exception as e:
            print(f"Warning: Could not set robot state: {e}")
    
    def _set_goal_state(self, state: np.ndarray):
        """Set goal state using robot's set_goal method."""
        try:
            if hasattr(self.robot, 'set_goal'):
                self.robot.set_goal(state[:3])
        except Exception as e:
            print(f"Warning: Could not set goal state: {e}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Set numpy random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        custom_behavior.set_gym_action(0.0, 0.0)
        
        # Sample robot position
        if self.randomize_start:
            new_robot_pos = self._sample_valid_robot_position()
        else:
            new_robot_pos = self.default_robot_state.copy()
        
        # Sample goal position (with distance constraint from robot)
        if self.randomize_goal:
            new_goal_pos = self._sample_valid_goal_position(new_robot_pos)
        else:
            new_goal_pos = self.default_goal_state.copy()
        
        # Reset IRSIM environment
        if hasattr(self.irsim_env, 'reset'):
            self.irsim_env.reset()
        
        # Set positions
        self._set_robot_state(new_robot_pos)
        self._set_goal_state(new_goal_pos)
        
        # Reset flags
        if hasattr(self.robot, 'collision_flag'):
            self.robot.collision_flag = False
        if hasattr(self.robot, 'arrive_flag'):
            self.robot.arrive_flag = False
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        """Execute one timestep."""
        custom_behavior.set_gym_action(float(action[0]), float(action[1]))
        self.irsim_env.step()
        
        info = self._get_info()
        
        # Calculate reward
        if info["goal_reached"]:
            reward = 100.0
        elif info["collision"]:
            reward = -50.0
        else:
            reward = -info["distance_to_goal"] * 0.1
        
        terminated = info["goal_reached"] or info["collision"]
        
        return self._get_obs(), reward, terminated, False, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self.irsim_env.render()
    
    def close(self):
        """Close the environment."""
        if hasattr(self.irsim_env, 'end'):
            self.irsim_env.end()


# Register the environment
gym.register(
    id="gymnasium_env/IRSIM-v0",
    entry_point=IRSIMEnv,
    max_episode_steps=300
)