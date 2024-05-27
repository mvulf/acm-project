# import regelum

from regelum.observer import Observer
from regelum.utils import rg

from src.system import HydraulicSystem

class HydraulicObserver(Observer):
    
    def __init__(
        self,
        system: HydraulicSystem
    ):
        self.x_p_init = system.init_state[0]
        self.x_th_limits = system._parameters["x_th_limits"]
        self.D_work_exit_2_ratio = system._parameters["D_work_exit_2_ratio"]
    
    def get_state_estimation(self, t, observation, action):
        observation = observation.flatten()
        # Get observation parameters
        x_jet, v_jet, p_hydr, p_work, x_th = [
            observation[i] for i in range(len(observation))
        ]
        # # Get throttle position in assumption that it is equal action
        # x_th = action
        # x_th_limits = self.x_th_limits
        # x_th = rg.if_else(x_th > x_th_limits[0], x_th, x_th_limits[0])
        # x_th = rg.if_else(x_th < x_th_limits[1], x_th, x_th_limits[1])
        
        x_p = 1e3*x_jet/self.D_work_exit_2_ratio + self.x_p_init
        v_p = 1e3*v_jet/self.D_work_exit_2_ratio
        
        state_estimation = rg.array(
            [[x_p, v_p, x_th, p_hydr, p_work]],
            # [x_p, v_p, x_th, p_hydr, p_work],
            prototype=observation
        )
        
        return state_estimation