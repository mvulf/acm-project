import numpy as np

from regelum.policy import Policy
from regelum.utils import rg

# from src._system import HydraulicSystemSimpleRg


def get_relative_observation(observation, l_crit:float, sampling_time:float):
    relative_observation = rg.zeros(
        observation.shape,
        prototype=observation,
    )
    
    relative_observation[:,0] = observation[:,0] / l_crit
    relative_observation[:,1] = observation[:,1] / l_crit * sampling_time
    
    return relative_observation


class PDController(Policy):
    def __init__(
        self,
        system, # TODO: CHANGE!
        sampling_time: float,
        P_coef:float=25.,
        D_coef:float=0.,
    ):
        super().__init__()
        self.system = system
        self.sampling_time = sampling_time
        
        self.pd_coefs: list[float] = [
            P_coef,
            D_coef,
        ]
        
    def get_action(self, observation):
        
        relative_observation = get_relative_observation(
            observation=observation,
            l_crit=self.system._parameters["l_crit"],
            sampling_time=self.sampling_time,
        )
        
        action = (
            self.pd_coefs[0] * (1 - relative_observation[:,0]) 
            - self.pd_coefs[1] * relative_observation[:,1]
        )
        
        return np.array([action])