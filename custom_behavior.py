import numpy as np
from irsim.lib import register_behavior

# Global variable to store the current action from Gymnasium
_current_action = np.array([[0.0], [0.0]])

def set_gym_action(linear_vel, angular_vel):
    """Set the action from the Gymnasium environment."""
    global _current_action
    _current_action = np.array([[float(linear_vel)], [float(angular_vel)]])

def get_gym_action():
    """Get the current action."""
    global _current_action
    return _current_action.copy()

@register_behavior("diff", "gym_control")
def beh_gym_control(ego_object, external_objects=None, **kwargs):
    """
    Custom behavior for Gymnasium control.
    This behavior returns the velocity command set by the Gymnasium environment.
    """
    return get_gym_action()