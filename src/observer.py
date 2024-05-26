# import regelum

from regelum.observer import Observer
from regelum.utils import rg

from src.system import HydraulicSystem


class HydraulicStationaryObserver(Observer):
    def __init__(
        self,
        system: HydraulicSystem
    ):
        self.x_p_init = system.init_state[0]
        self.x_th_limits = system._parameters["x_th_limits"]
        self.D_work_exit_2_ratio = system._parameters["D_work_exit_2_ratio"]
        
        self.system = system
        
        
    def get_state_estimation(self, t, observation, action):
        observation = observation.flatten()
        # Get observation parameters
        x_jet, v_jet = observation[0], observation[1]
        
        x_p = 1e3*x_jet/self.D_work_exit_2_ratio + self.x_p_init
        v_p = 1e3*v_jet/self.D_work_exit_2_ratio
        
        # ADD x_th estimation
        A_hydr, A_work = self.system._parameters["A_hydr"], self.system._parameters["A_work"]
        
        # Required dynamic parameters
        F_coulomb, eta, F_g, g, m_p = (
            self.system._parameters["F_coulomb"],
            self.system._parameters["eta"],
            self.system._parameters["F_g"],
            self.system._parameters["g"],
            self.system._parameters["m_p"],
        )
        
        p_l, B_th, K_hydr = (
            self.system._parameters["p_l"],
            self.system._parameters["B_th"],
            self.system._parameters["K_hydr"],
        )
        
        p_atm, B_exit, K_work, h_work_init = (
            self.system._parameters["p_atm"],
            self.system._parameters["B_exit"],
            self.system._parameters["K_work"],
            self.system._parameters["h_work_init"],
        )
        
        F_hydr = m_p*g/(rg.sign(v_p)*(1-eta)-1)
        
        F_hydr = rg.if_else(
            (1-eta)*F_hydr>F_coulomb,
            F_hydr,
            rg.sign(v_p)*F_coulomb - m_p*g
        )
        
        # Pressure estimation by piston velocity
        p_hydr_v_p = (F_hydr + A_work*((v_p/B_exit)**2 + p_atm))/A_hydr
        # Throttle position estimation
        x_th = (
            v_p
            /(rg.sign(p_l - p_hydr_v_p)*B_th*(rg.abs(p_l - p_hydr_v_p))**(1/2))
        )
        
        state_estimation = rg.array(
            [[x_p, v_p, x_th]],
            prototype=observation
        )
        # state_estimation = rg.array(
        #     [[x_p]],
        #     prototype=observation
        # )
        
        return state_estimation


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
        x_jet, v_jet, p_hydr, p_work = [
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
            [[x_p, v_p, p_hydr, p_work]],
            # [x_p, v_p, x_th, p_hydr, p_work],
            prototype=observation
        )
        
        return state_estimation