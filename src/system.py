import numpy as np
from typing import Tuple, Dict, Optional, Callable, Type, Any

from regelum.system import System
from regelum.utils import rg

class HydraulicSystem(System):
    """System class: hydraulic system. State transition function"""
    
    _name = 'HydraulicSystem'
    _system_type = 'diff_eqn'
    _dim_state = 3
    _dim_inputs = 1
    _dim_observation = 2
    _state_naming = ["piston_position", "piston_velocity", "throttle_position"]
    _observation_naming = ["jet_length", "jet_velocity"]
    _inputs_naming = ["throttle_action"]
    _action_bounds = [[-20.0, 20.0]]