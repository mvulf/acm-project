from casadi.casadi import MX
import numpy as np
from typing import Tuple, Dict, Optional, Callable, Type, Any

from regelum.system import System
from regelum.utils import rg

class HydraulicSystemFull(System):
    """System class: hydraulic system. State transition function"""
    
    _name = 'HydraulicSystemFull'
    _system_type = 'diff_eqn'
    _dim_state = 3
    _dim_inputs = 1
    _dim_observation = 2
    _state_naming = [
        "piston position [µm]", 
        "piston velocity [µm/s]", 
        "throttle position [µm]"
    ]
    _observation_naming = ["jet length [mm]", "jet velocity [mm/s]"]
    _inputs_naming = ["throttle_action"]
    _action_bounds = [[-20.0, 20.0]]
     
    def __init__(
        self,
        *args,
        system_parameters_init = {
            "p_l_gauge": 1.5e5,
            "p_hydr_init_gauge": 0.,
            "x_th_limits": (0., 20.),
            "freq_th": 500.0,
            "m_p": 20e-3,
            "D_th": 200e-6,
            "D_hydr": 20e-3,
            "D_work": 20e-3,
            "D_exit": 0.33e-3,
            "l_exit": 8.5e-3,
            "p_c": 10e3,
            "eta": 0.70,
            "zeta_th": 5.0,
            "rho_hydr": 1e3,
            "rho_work": 1e3,
            "beta_v_hydr": 0.49e-9,
            "beta_v_work": 0.49e-9,
            "sigma_work": 73e-3,
            "mu_work": 1.0e-3,
            "v_j": 200.,
            "jet_length_std": 5e-2, # TODO: find absolute std
            "jet_velocity_std": 1e-2, # TODO: find absolute std
            "p_atm": 1e5, # Atmosphere (ambient) pressure, Pa
            "g": 9.81, # gravity constant, m/s^2
            "x_th_eps": 0.5, # backlash (should be as small as possible)
            "dx_th_eps": 0.1, # used as limit for positions checking
        },
        **kwargs
    ):
        super().__init__(
            *args,
            system_parameters_init=system_parameters_init,
            **kwargs
        )
        """Droplet generator (hydraulic) system

        Args:
            system_parameters_init: parameters of the system:
                p_l_gauge: Gauge liquid pressure before throttle [Pa]. 
                    Defaults to 1.5e5.
                p_hydr_init_gauge: Gauge hydraulic container pressure [Pa]. 
                    Defaults to 0..
                x_th_limits: Real throttle position limits [µm]. 
                    Defaults to (0, 20).
                freq_th: Frottle frequency [Hz]. Defaults to 500.0.
                m_p: Piston mass [kg]. Defaults to 20e-3.
                D_th: Equivalent throttle diameter [m]. Defaults to 200e-6.
                D_hydr: Hydraulic container diameter [m]. Defaults to 20e-3.
                D_work: Working container diameter [m]. Defaults to 20e-3.
                D_exit: Exit orifice diameter [m]. Defaults to 0.33e-3.
                l_exit: Exit orifice length [m]. Defaults to 8.5e-3.
                p_c: Pressure difference on the piston to start movement [Pa].
                    Defaults to 10e3.
                eta: Mechanical efficiency. Defaults to 0.70.
                zeta_th: Hydraulic throttle coefficient. 
                    Might be find empirically (from real equipment). 
                    Now it is taken for the valve, see 
                    'Идельчик И. Е. Справочник по гидравлическим 
                    сопротивлениям. М., "Машиностроение", 1975'. 
                    Defaults to 5.0.
                rho_hydr: Hydraulic liquid density [kg/m^3]. Defaults to 1e3.
                rho_work: Working liquid density [kg/m^3]. Defaults to 1e3.
                beta_v_hydr: Hydraulic liquid compressibility [Pa^-1]. 
                    Defaults to 0.49e-9.
                beta_v_work: Working liquid compressibility [Pa^-1]. 
                    Defaults to 0.49e-9.
                sigma_work: Working liquid surface tension [N/m]. 
                    Defaults to 73e-3.
                mu_work: Working liquid viscosity[Pa*s]. Defaults to 1.0e-3.
                v_j: Jet speed for the stable operation (found experimentaly)
                    [mm/s]. Defaults to 200.
                jet_length_std: Standard deviation of Relative jet length 
                    observation. Defaults to 5e-2.
                jet_velocity_std: Standard deviation of Relative jet velocity 
                    observation. Defaults to 1e-2.
                p_atm: Atmosphere (ambient) pressure, Pa. 
                    Defaults to 1e5.
                g: Gravity constant, m/s^2. Defaults to 9.81. 
                x_th_eps: Backlash (should be as small as possible). 
                    Defaults to 0.5.
                dx_th_eps: Used as limit for positions checking.
                    Defaults to 0.1.
        """
        
        # Pressure and Throttle parameters
        p_atm, p_l_gauge, p_hydr_init_gauge = (
            self._parameters["p_atm"],
            self._parameters["p_l_gauge"],
            self._parameters["p_hydr_init_gauge"],
        )
        freq_th, x_th_limits = (
            self._parameters["freq_th"],
            self._parameters["x_th_limits"]
        )
        self.update_system_parameters(
            {
                "p_l": p_atm + p_l_gauge,
                "p_hydr_init": p_atm + p_hydr_init_gauge,
                # Max throttle speed, µm/s.
                "v_th_max": freq_th*(x_th_limits[1] - x_th_limits[0]),
            }
        )
        
        # Piston mass, gravity parameters; Gravity force
        m_p, g = (
            self._parameters["m_p"],
            self._parameters["g"]
        )
        self.update_system_parameters(
            {
                "F_g": m_p*g, # Gravity force, N
            }
        )

        # Geometry parameters        
        D_th, D_hydr, D_work, D_exit, l_exit = (
            self._parameters["D_th"],
            self._parameters["D_hydr"],
            self._parameters["D_work"],
            self._parameters["D_exit"],
            self._parameters["l_exit"],
        )
        # FUNCTION GET AREA BY DIAMETER
        # returns area in [m^2], if input in [m]
        get_area = lambda D: np.pi*D**2/4
        self.update_system_parameters(
            {
                "A_hydr": get_area(D_hydr), # m^2
                "A_work": get_area(D_work), # m^2
                "D_work_exit_2_ratio": D_work**2/D_exit**2,
            }
        )
        
        # Friction force
        A_hydr, A_work = self._parameters["A_hydr"], self._parameters["A_work"]
        A_max = max(A_hydr, A_work) # m^2
        # Coulomb friction force, N
        self.update_system_parameters(
            {
                "F_c": self._parameters["p_c"] * A_max
            }
        )
        # Exit hydraulic coefficient
        # see https://doi.org/10.1201/9781420040470
        C_D = 0.827 - 0.0085*l_exit/D_exit
        self.update_system_parameters(
            {
                "zeta_exit": 1/C_D**2
            }
        )
        
        # Liquid params
        rho_hydr, rho_work, sigma_work, mu_work = (
            self._parameters["rho_hydr"],
            self._parameters["rho_work"],
            self._parameters["sigma_work"],
            self._parameters["mu_work"],
        )
        
        # PRESSURE LOSSES
        zeta_th, zeta_exit = (
            self._parameters["zeta_th"], self._parameters["zeta_exit"]
        )
        self.update_system_parameters(
            {
                # Capillar pressure difference to othercome for drop exiting
                "p_capillar_max": 4*sigma_work/D_exit,
                "ploss_coef_hydr": (zeta_th*rho_hydr*D_hydr**4)/(32*D_th**2),
                "ploss_coef_work": (zeta_exit*rho_work*D_work**4)/\
                    (2e12*D_exit**4),
            }
        )
        
        # Dimensionless jet numbers
        v_j = self._parameters["v_j"]
        We_j = rho_work*v_j**2*D_exit/(1e6*sigma_work)
        Re_j = rho_work*v_j*D_exit/(1e3*mu_work)
        Oh_j = np.sqrt(We_j)/Re_j
         
        # JET LENGTH AND DROP DIAMETER
        # Critical jet length
        # see https://doi.org/10.1007/s00348-003-0629-6
        l_crit = 13.4e3*(np.sqrt(We_j)\
            + 3*We_j/Re_j) * D_exit
        # Estimated Droplet diameter
        D_drop = 1e3*(1.5*np.pi*np.sqrt(2 + 3*Oh_j))**(1/3) * D_exit
        self.update_system_parameters(
            {
                "l_crit": l_crit,
                "D_drop": D_drop,
            }
        )
        
        # RESET piston position and last pressure in hydraulic container
        self.reset()
        
        
    def reset(self) -> None:
        """Reset system to initial state."""
        self.update_system_parameters(
            {
                # p_h|_{x_{th}>0}
                "_p_hydr_last": self._parameters["p_hydr_init"],
                # x_p|_{x_{th}>0}
                "_x_p_last": None, # define later, if None
                # Initial piston position
                "_x_p_init": None, # define later, if None
            }
        )


    def get_pressure_hydraulic(self, state) -> float:
        """ Get pressure in the hydraulic container

        Args:
            state: array of current state:
                x_p (float): piston position [µm]
                v_p (float): piston velocity [µm/s]
                x_th (float): throttle position [µm]

        Returns:
            float: pressure in the hydraulic container [Pa]
        """
        
        # State params
        x_p, v_p, x_th = state[0], state[1], state[2]
        
        # Define last piston position first time as init piston position
        if self._parameters["_x_p_last"] is None:
            self.update_system_parameters(
                {
                    "_x_p_last": x_p,
                }
            )
        
        # System params
        p_l, x_th_eps, ploss_coef_hydr, p_hydr_last, x_p_last, beta_v_hydr = (
            self._parameters["p_l"],
            self._parameters["x_th_eps"],
            self._parameters["ploss_coef_hydr"],
            self._parameters["_p_hydr_last"],
            self._parameters["_x_p_last"],
            self._parameters["beta_v_hydr"],
        )
        
        # Calculate
        if x_th > 0:
            pressure_hydraulic = p_l
            # dynamic pressure loss happends only when there is a flow rate
            if v_p != 0: 
                # self.x_th_eps refers to somekind of backslash
                pressure_hydraulic -= v_p*(abs(v_p)/\
                    max(x_th_eps, x_th)**2)*ploss_coef_hydr
            
            # Save piston position and hydraulic pressure if throttle is opened
            self.update_system_parameters(
                {
                    "_x_p_last": x_p,
                    "_p_hydr_last": pressure_hydraulic,
                }
            )
        else:
            # if x_p < 0:
            #     print('WARNING: piston position might be positive!')
            # assert x_p > 0, 'piston position might be positive'
            pressure_hydraulic = p_hydr_last +\
                (x_p_last/x_p - 1)/beta_v_hydr

        return pressure_hydraulic
    
    
    def get_pressure_working(self, state) -> float:
        """ Get pressure in the working container

        Args:
            state: array of current state:
                x_p (float): piston position [µm]
                v_p (float): piston velocity [µm/s]
                x_th (float): throttle position [µm] (Do not required)

        Returns:
            float: pressure in the test container [Pa]
        """
        # State params
        x_p, v_p = state[0], state[1]

        # Define init piston position
        if self._parameters["_x_p_init"] is None:
            self.update_system_parameters(
                {
                    "_x_p_init": x_p,
                }
            )
        
        # Required parameters
        x_p_init, p_capillar_max, beta_v_work, p_atm, ploss_coef_work = (
            self._parameters["_x_p_init"],
            self._parameters["p_capillar_max"],
            self._parameters["beta_v_work"],
            self._parameters["p_atm"],
            self._parameters["ploss_coef_work"],
        )
        
        # Position difference
        dx_p = x_p - x_p_init
        
        p_capillar = min(
            p_capillar_max, 
            abs(dx_p/x_p)/beta_v_work
        )
        
        pressure_working = p_atm + np.sign(dx_p) * p_capillar
            
        # dynamic pressure loss happends only when there is a flow rate
        if v_p != 0:
            pressure_working += v_p*abs(v_p) * ploss_coef_work
        
        return pressure_working
    
    
    def get_force_hydraulic(self, state) -> float:
        """ Get hydraulic force acting on the piston

        Args:
            state: array of current state:
                x_p (float): piston position [µm] (Do not required)
                v_p (float): piston velocity [µm/s]
                x_th (float): throttle position [µm] (Do not required)

        Returns:
            float: hydraulic force [N]
        """
        
        p_hydr = self.get_pressure_hydraulic(state)
        p_work = self.get_pressure_working(state)
        
        A_hydr, A_work = self._parameters["A_hydr"], self._parameters["A_work"]
        
        return A_hydr*p_hydr - A_work*p_work
    
    
    def get_force_friction(self, state, F_h: float) -> float:
        """ Get friction force acting on the piston

        Args:
            state:
            v_p (float): array of current state:
                x_p (float): piston position [µm]. Do not required
                v_p (float): piston velocity [µm/s]
                x_th (float): throttle position [µm]. Do not required
            F_h (float): Hydraulic force [N]

        Returns:
            float: friction force [N]
        """
        
        v_p = state[1]
        
        # Required parameters
        F_c, eta, F_g = (
            self._parameters["F_c"],
            self._parameters["eta"],
            self._parameters["F_g"],
        )
        
        if v_p != 0:
            return - np.sign(v_p) * max(F_c, (1-eta)*F_h)
        # If piston does not move
        return -np.sign(F_g + F_h) * F_c
    
    
    def get_acceleration(self, state) -> float:
        """ Get piston acceleration

        Args:
            state: array of current state:
                x_p (float): piston position [µm]
                v_p (float): piston velocity [µm/s]
                x_th (float): throttle position [µm]

        Returns:
            float: piston acceleration [m/s^2]
        """
        
        # State params
        v_p = state[1]
        
        F_h = self.get_force_hydraulic(state)
        F_fr = self.get_force_friction(state, F_h)
        
        # Required params
        F_g, g, m_p = (
            self._parameters["F_g"],
            self._parameters["g"],
            self._parameters["m_p"],
        )
        
        if (abs(v_p) > 0) or (abs(F_h + F_g) > abs(F_fr)):
            return (g + 1/m_p * (F_h + F_fr))*1e6
        return 0 # if piston does not move and acting force lower than friction
    
    
    def _compute_state_dynamics(self, time, state, inputs):
        
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )
        
        x_th_limits = self._parameters["x_th_limits"]
        
        # real throttle position
        # If real throttle position out of bounds - 
        # end throttle movement and set in bounds
        x_th = rg.clip(state[2], x_th_limits[0], x_th_limits[1])
        x_th_act = rg.clip(inputs[0], x_th_limits[0], x_th_limits[1])
        
        # \dot{x_p}
        Dstate[0] = state[1]
        # \dot{v_p}
        Dstate[1] = self.get_acceleration(state)
        
        # TODO: make First-order aperiodic chain: 1/T * (k*x - y)
        dx_th_eps, v_th_max = (
            self._parameters["dx_th_eps"],
            self._parameters["v_th_max"]
        )
        # \dot{x_th}
        # if real throttle position is differ from the set one, change it
        if abs(x_th_act - x_th) > dx_th_eps:
            Dstate[2] = np.sign(x_th_act - x_th) * v_th_max
        else:
            x_th = x_th_act # set throttle position exact as what we want to act
            state[2] = x_th # Set exact position
            Dstate[2] = 0
            
        return Dstate
    
    def compute_closed_loop_rhs(self, time, state):
        return self._compute_state_dynamics(time, state, self.inputs)
    
    
    def get_clean_observation(self, state):
        """Get clean observations 
        (relative jet length and relative jet velocity), without sensors noise

        Args:
            state: system state

        Returns:
            observation (jet length, jet velocity)
        """
        x_p = state[0]
        v_p = state[1]
        
        observation = rg.zeros(
            self.dim_observation,
            prototype=state,
        )
        
        # Define init piston position
        if self._parameters["_x_p_init"] is None:
            self.update_system_parameters(
                {
                    "_x_p_init": x_p,
                }
            )
        
        x_p_init, D_work_exit_2_ratio = (
            self._parameters["_x_p_init"],
            self._parameters["D_work_exit_2_ratio"],
        )
        
        # Jet length
        observation[0] = (
            1e-3 * (x_p - x_p_init) * D_work_exit_2_ratio
        )
        
        # Jet velocity
        observation[1] = (
            1e-3 * v_p * D_work_exit_2_ratio
        )
        
        return observation
    

    def _get_observation(self, time, state, inputs):
        """ Get observation with normal noise
        """
        observation = self.get_clean_observation(state)
        
        # relative jet length with noise
        observation[0] += np.random.normal(
            scale=self._parameters["jet_length_std"]
        )
        # relative jet velocity with noise
        observation[1] += np.random.normal(
            scale=self._parameters["jet_velocity_std"]
        )

        
        return observation
    

class HydraulicSystemFullRg(System):
    """System class: hydraulic system. State transition function"""
    
    _name = 'HydraulicSystemFullRg'
    _system_type = 'diff_eqn'
    _dim_state = 3
    _dim_inputs = 1
    _dim_observation = 2
    _state_naming = [
        "piston position [µm]", 
        "piston velocity [µm/s]", 
        "throttle position [µm]"
    ]
    _observation_naming = ["jet length [mm]", "jet velocity [mm/s]"]
    _inputs_naming = ["throttle_action"]
    _action_bounds = [[-20.0, 20.0]]
    
    def __init__(
        self,
        *args,
        system_parameters_init = {
            "p_l_gauge": 1.5e5,
            "p_hydr_init_gauge": 0.,
            "x_th_limits": (0., 20.),
            "freq_th": 500.0,
            "m_p": 20e-3,
            "D_th": 200e-6,
            "D_hydr": 20e-3,
            "D_work": 20e-3,
            "D_exit": 0.33e-3,
            "l_exit": 8.5e-3,
            "p_c": 10e3,
            "eta": 0.70,
            "zeta_th": 5.0,
            "rho_hydr": 1e3,
            "rho_work": 1e3,
            "beta_v_hydr": 0.49e-9,
            "beta_v_work": 0.49e-9,
            "sigma_work": 73e-3,
            "mu_work": 1.0e-3,
            "v_j": 200.,
            "jet_length_std": 5e-2, # TODO: find absolute std
            "jet_velocity_std": 1e-2, # TODO: find absolute std
            "p_atm": 1e5, # Atmosphere (ambient) pressure, Pa
            "g": 9.81, # gravity constant, m/s^2
            "x_th_eps": 0.5, # backlash (should be as small as possible)
            "dx_th_eps": 0.1, # used as limit for positions checking
        },
        **kwargs
    ):
        super().__init__(
            *args,
            system_parameters_init=system_parameters_init,
            **kwargs
        )
        """Droplet generator (hydraulic) system

        Args:
            system_parameters_init: parameters of the system:
                p_l_gauge: Gauge liquid pressure before throttle [Pa]. 
                    Defaults to 1.5e5.
                p_hydr_init_gauge: Gauge hydraulic container pressure [Pa]. 
                    Defaults to 0..
                x_th_limits: Real throttle position limits [µm]. 
                    Defaults to (0, 20).
                freq_th: Frottle frequency [Hz]. Defaults to 500.0.
                m_p: Piston mass [kg]. Defaults to 20e-3.
                D_th: Equivalent throttle diameter [m]. Defaults to 200e-6.
                D_hydr: Hydraulic container diameter [m]. Defaults to 20e-3.
                D_work: Working container diameter [m]. Defaults to 20e-3.
                D_exit: Exit orifice diameter [m]. Defaults to 0.33e-3.
                l_exit: Exit orifice length [m]. Defaults to 8.5e-3.
                p_c: Pressure difference on the piston to start movement [Pa].
                    Defaults to 10e3.
                eta: Mechanical efficiency. Defaults to 0.70.
                zeta_th: Hydraulic throttle coefficient. 
                    Might be find empirically (from real equipment). 
                    Now it is taken for the valve, see 
                    'Идельчик И. Е. Справочник по гидравлическим 
                    сопротивлениям. М., "Машиностроение", 1975'. 
                    Defaults to 5.0.
                rho_hydr: Hydraulic liquid density [kg/m^3]. Defaults to 1e3.
                rho_work: Working liquid density [kg/m^3]. Defaults to 1e3.
                beta_v_hydr: Hydraulic liquid compressibility [Pa^-1]. 
                    Defaults to 0.49e-9.
                beta_v_work: Working liquid compressibility [Pa^-1]. 
                    Defaults to 0.49e-9.
                sigma_work: Working liquid surface tension [N/m]. 
                    Defaults to 73e-3.
                mu_work: Working liquid viscosity[Pa*s]. Defaults to 1.0e-3.
                v_j: Jet speed for the stable operation (found experimentaly)
                    [mm/s]. Defaults to 200.
                jet_length_std: Standard deviation of Relative jet length 
                    observation. Defaults to 5e-2.
                jet_velocity_std: Standard deviation of Relative jet velocity 
                    observation. Defaults to 1e-2.
                p_atm: Atmosphere (ambient) pressure, Pa. 
                    Defaults to 1e5.
                g: Gravity constant, m/s^2. Defaults to 9.81. 
                x_th_eps: Backlash (should be as small as possible). 
                    Defaults to 0.5.
                dx_th_eps: Used as limit for positions checking.
                    Defaults to 0.1.
        """
        
        # Pressure and Throttle parameters
        p_atm, p_l_gauge, p_hydr_init_gauge = (
            self._parameters["p_atm"],
            self._parameters["p_l_gauge"],
            self._parameters["p_hydr_init_gauge"],
        )
        freq_th, x_th_limits = (
            self._parameters["freq_th"],
            self._parameters["x_th_limits"]
        )
        self.update_system_parameters(
            {
                "p_l": p_atm + p_l_gauge,
                "p_hydr_init": p_atm + p_hydr_init_gauge,
                # Max throttle speed, µm/s.
                "v_th_max": freq_th*(x_th_limits[1] - x_th_limits[0]),
            }
        )
        
        # Piston mass, gravity parameters; Gravity force
        m_p, g = (
            self._parameters["m_p"],
            self._parameters["g"]
        )
        self.update_system_parameters(
            {
                "F_g": m_p*g, # Gravity force, N
            }
        )

        # Geometry parameters        
        D_th, D_hydr, D_work, D_exit, l_exit = (
            self._parameters["D_th"],
            self._parameters["D_hydr"],
            self._parameters["D_work"],
            self._parameters["D_exit"],
            self._parameters["l_exit"],
        )
        # FUNCTION GET AREA BY DIAMETER
        # returns area in [m^2], if input in [m]
        get_area = lambda D: np.pi*D**2/4
        self.update_system_parameters(
            {
                "A_hydr": get_area(D_hydr), # m^2
                "A_work": get_area(D_work), # m^2
                "D_work_exit_2_ratio": D_work**2/D_exit**2,
            }
        )
        
        # Friction force
        A_hydr, A_work = self._parameters["A_hydr"], self._parameters["A_work"]
        A_max = max(A_hydr, A_work) # m^2
        # Coulomb friction force, N
        self.update_system_parameters(
            {
                "F_c": self._parameters["p_c"] * A_max
            }
        )
        # Exit hydraulic coefficient
        # see https://doi.org/10.1201/9781420040470
        C_D = 0.827 - 0.0085*l_exit/D_exit
        self.update_system_parameters(
            {
                "zeta_exit": 1/C_D**2
            }
        )
        
        # Liquid params
        rho_hydr, rho_work, sigma_work, mu_work = (
            self._parameters["rho_hydr"],
            self._parameters["rho_work"],
            self._parameters["sigma_work"],
            self._parameters["mu_work"],
        )
        
        # PRESSURE LOSSES
        zeta_th, zeta_exit = (
            self._parameters["zeta_th"], self._parameters["zeta_exit"]
        )
        self.update_system_parameters(
            {
                # Capillar pressure difference to othercome for drop exiting
                "p_capillar_max": 4*sigma_work/D_exit,
                "ploss_coef_hydr": (zeta_th*rho_hydr*D_hydr**4)/(32*D_th**2),
                "ploss_coef_work": (zeta_exit*rho_work*D_work**4)/\
                    (2e12*D_exit**4),
            }
        )
        
        # Dimensionless jet numbers
        v_j = self._parameters["v_j"]
        We_j = rho_work*v_j**2*D_exit/(1e6*sigma_work)
        Re_j = rho_work*v_j*D_exit/(1e3*mu_work)
        Oh_j = np.sqrt(We_j)/Re_j
         
        # JET LENGTH AND DROP DIAMETER
        # Critical jet length
        # see https://doi.org/10.1007/s00348-003-0629-6
        l_crit = 13.4e3*(np.sqrt(We_j)\
            + 3*We_j/Re_j) * D_exit
        # Estimated Droplet diameter
        D_drop = 1e3*(1.5*np.pi*np.sqrt(2 + 3*Oh_j))**(1/3) * D_exit
        self.update_system_parameters(
            {
                "l_crit": l_crit,
                "D_drop": D_drop,
            }
        )
        
        # RESET piston position and last pressure in hydraulic container
        self.reset()
        
        
    def reset(self) -> None:
        """Reset system to initial state."""
        self.update_system_parameters(
            {
                # p_h|_{x_{th}>0}
                "_p_hydr_last": self._parameters["p_hydr_init"],
                # x_p|_{x_{th}>0}
                "_x_p_last": None, # define later, if None
                # Initial piston position
                "_x_p_init": None, # define later, if None
            }
        )


    def get_pressure_hydraulic(self, state) -> float:
        """ Get pressure in the hydraulic container

        Args:
            state: array of current state:
                x_p (float): piston position [µm]
                v_p (float): piston velocity [µm/s]
                x_th (float): throttle position [µm]

        Returns:
            float: pressure in the hydraulic container [Pa]
        """
        
        # State params
        x_p, v_p, x_th = state[0], state[1], state[2]
        
        # Define last piston position first time as init piston position
        if self._parameters["_x_p_last"] is None:
            self.update_system_parameters(
                {
                    "_x_p_last": x_p,
                }
            )
        
        # System params
        p_l, x_th_eps, ploss_coef_hydr, p_hydr_last, x_p_last, beta_v_hydr = (
            self._parameters["p_l"],
            self._parameters["x_th_eps"],
            self._parameters["ploss_coef_hydr"],
            self._parameters["_p_hydr_last"],
            self._parameters["_x_p_last"],
            self._parameters["beta_v_hydr"],
        )

        pressure_hydraulic_open = (
            p_l
            - v_p*ploss_coef_hydr*rg.abs(v_p)/\
                (rg.if_else(x_th_eps > x_th, x_th_eps, x_th)**2)
        )
        pressure_hydraulic_closed = (
            p_hydr_last + (x_p_last/x_p - 1)/beta_v_hydr
        )

        pressure_hydraulic_open = rg.if_else(
            v_p != 0, pressure_hydraulic_open, p_l
        )

        pressure_hydraulic = rg.if_else(
            x_th > 0, pressure_hydraulic_open, pressure_hydraulic_closed
        )

        # DOES NOT WORK WITH CASADI
        # Save piston position and hydraulic pressure if throttle is opened
        if x_th > 0:
            self.update_system_parameters(
                {
                    "_x_p_last": x_p,
                    "_p_hydr_last": pressure_hydraulic,
                }
            )

        return pressure_hydraulic
    
    
    def get_pressure_working(self, state) -> float:
        """ Get pressure in the working container

        Args:
            state: array of current state:
                x_p (float): piston position [µm]
                v_p (float): piston velocity [µm/s]
                x_th (float): throttle position [µm] (Do not required)

        Returns:
            float: pressure in the test container [Pa]
        """
        # State params
        x_p, v_p = state[0], state[1]

        # Define init piston position
        if self._parameters["_x_p_init"] is None:
            self.update_system_parameters(
                {
                    "_x_p_init": x_p,
                }
            )
        
        # Required parameters
        x_p_init, p_capillar_max, beta_v_work, p_atm, ploss_coef_work = (
            self._parameters["_x_p_init"],
            self._parameters["p_capillar_max"],
            self._parameters["beta_v_work"],
            self._parameters["p_atm"],
            self._parameters["ploss_coef_work"],
        )
        
        # Position difference
        dx_p = x_p - x_p_init

        p_compress = rg.abs(dx_p/x_p)/beta_v_work

        p_capillar = rg.if_else(
            p_capillar_max < p_compress, p_capillar_max, p_compress
        )

        pressure_working = p_atm + rg.sign(dx_p) * p_capillar

        pressure_working = rg.if_else(
            v_p != 0,
            pressure_working + v_p*rg.abs(v_p)*ploss_coef_work,
            pressure_working
        )

        return pressure_working
    
    
    def get_force_hydraulic(self, state) -> float:
        """ Get hydraulic force acting on the piston

        Args:
            state: array of current state:
                x_p (float): piston position [µm] (Do not required)
                v_p (float): piston velocity [µm/s]
                x_th (float): throttle position [µm] (Do not required)

        Returns:
            float: hydraulic force [N]
        """
        
        p_hydr = self.get_pressure_hydraulic(state)
        p_work = self.get_pressure_working(state)
        
        A_hydr, A_work = self._parameters["A_hydr"], self._parameters["A_work"]        
        return A_hydr*p_hydr - A_work*p_work
    
    
    def get_force_friction(self, state, F_h: float) -> float:
        """ Get friction force acting on the piston

        Args:
            state:
            v_p (float): array of current state:
                x_p (float): piston position [µm]. Do not required
                v_p (float): piston velocity [µm/s]
                x_th (float): throttle position [µm]. Do not required
            F_h (float): Hydraulic force [N]

        Returns:
            float: friction force [N]
        """
        
        v_p = state[1]
        
        # Required parameters
        F_c, eta, F_g = (
            self._parameters["F_c"],
            self._parameters["eta"],
            self._parameters["F_g"],
        )
        
        F_fr_h = (1-eta)*F_h
        
        F_fr_dynamic = -rg.sign(v_p) * rg.if_else(F_c > F_fr_h, F_c, F_fr_h)
        # If piston does not move
        F_fr_static = -rg.sign(F_g + F_h) * F_c
        
        F_fr = rg.if_else(v_p != 0, F_fr_dynamic, F_fr_static)

        return F_fr
    
    
    def get_acceleration(self, state) -> float:
        """ Get piston acceleration

        Args:
            state: array of current state:
                x_p (float): piston position [µm]
                v_p (float): piston velocity [µm/s]
                x_th (float): throttle position [µm]

        Returns:
            float: piston acceleration [m/s^2]
        """
        
        # State params
        v_p = state[1]
        
        F_h = self.get_force_hydraulic(state)
        F_fr = self.get_force_friction(state, F_h)
        
        # Required params
        F_g, g, m_p = (
            self._parameters["F_g"],
            self._parameters["g"],
            self._parameters["m_p"],
        )
        
        cond_velocity = (v_p != 0)
        cond_fr_overcome = (rg.abs(F_h + F_g) > rg.abs(F_fr))
        
        # return 0, if piston does not move and acting force lower than friction
        acceleration = rg.if_else(
            (cond_velocity + cond_fr_overcome) > 0, # OR
            1e6*(g + 1/m_p * (F_h + F_fr)),
            0
        )

        return acceleration
    
    
    def _compute_state_dynamics(self, time, state, inputs):
        
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )
        
        x_th_limits = self._parameters["x_th_limits"]

        x_th = rg.if_else(state[2] > x_th_limits[0], state[2], x_th_limits[0])
        x_th = rg.if_else(x_th < x_th_limits[1], x_th, x_th_limits[1])

        x_th_act = rg.if_else(
            inputs[0] > x_th_limits[0], inputs[0], x_th_limits[0]
        )
        x_th_act = rg.if_else(
            x_th_act < x_th_limits[1], x_th_act, x_th_limits[1]
        )
        
        # \dot{x_p}
        Dstate[0] = state[1]
        # \dot{v_p}
        Dstate[1] = self.get_acceleration(state)
        
        # TODO: make First-order aperiodic chain: 1/T * (k*x - y)
        dx_th_eps, v_th_max = (
            self._parameters["dx_th_eps"],
            self._parameters["v_th_max"]
        )
        # \dot{x_th}
        # if real throttle position is differ from the set one, change it

        cond_th_dyn = (rg.abs(x_th_act - x_th) > dx_th_eps)

        Dstate[2] = rg.if_else(
            cond_th_dyn, rg.sign(x_th_act - x_th) * v_th_max, 0
        )
        x_th = rg.if_else(cond_th_dyn, x_th, x_th_act)
        state[2] = rg.if_else(cond_th_dyn, state[2], x_th)

        return Dstate
    
    
    def get_clean_observation(self, state):
        """Get clean observations 
        (relative jet length and relative jet velocity), without sensors noise

        Args:
            state: system state

        Returns:
            observation (jet length, jet velocity)
        """
        x_p = state[0]
        v_p = state[1]
        
        observation = rg.zeros(
            self.dim_observation,
            prototype=state,
        )
        
        # Define init piston position
        if self._parameters["_x_p_init"] is None:
            self.update_system_parameters(
                {
                    "_x_p_init": x_p,
                }
            )
        
        x_p_init, D_work_exit_2_ratio = (
            self._parameters["_x_p_init"],
            self._parameters["D_work_exit_2_ratio"],
        )
        
        # Jet length
        observation[0] = (
            1e-3 * (x_p - x_p_init) * D_work_exit_2_ratio
        )
        
        # Jet velocity
        observation[1] = (
            1e-3 * v_p * D_work_exit_2_ratio
        )
        
        return observation
    

    def _get_observation(self, time, state, inputs):
        """ Get observation with normal noise
        """
        observation = self.get_clean_observation(state)
        
        # relative jet length with noise
        observation[0] += np.random.normal(
            scale=self._parameters["jet_length_std"]
        )
        # relative jet velocity with noise
        observation[1] += np.random.normal(
            scale=self._parameters["jet_velocity_std"]
        )
        
        return observation


class HydraulicSystemSimpleRg(System):
    """System class: hydraulic system. State transition function"""
    
    _name = 'HydraulicSystemFullRg'
    _system_type = 'diff_eqn'
    _dim_state = 3
    _dim_inputs = 1
    _dim_observation = 2
    _state_naming = [
        "piston position [µm]", 
        "piston velocity [µm/s]", 
        "throttle position [µm]"
    ]
    _observation_naming = ["jet length [mm]", "jet velocity [mm/s]"]
    _inputs_naming = ["throttle_action"]
    _action_bounds = [[-20.0, 20.0]]
    
    def __init__(
        self,
        *args,
        system_parameters_init = {
            "p_l_gauge": 1.5e5,
            "p_hydr_init_gauge": 0.,
            "x_th_limits": (0., 20.),
            "freq_th": 500.0,
            "m_p": 20e-3,
            "D_th": 200e-6,
            "D_hydr": 20e-3,
            "D_work": 20e-3,
            "D_exit": 0.33e-3,
            "l_exit": 8.5e-3,
            "p_c": 10e3,
            "eta": 0.70,
            "zeta_th": 5.0,
            "rho_hydr": 1e3,
            "rho_work": 1e3,
            # "beta_v_hydr": 0.49e-9,
            # "beta_v_work": 0.49e-9,
            "sigma_work": 73e-3,
            "mu_work": 1.0e-3,
            "v_j": 200.,
            # "jet_length_std": 5e-2, # TODO: find absolute std
            # "jet_velocity_std": 1e-2, # TODO: find absolute std
            "p_atm": 1e5, # Atmosphere (ambient) pressure, Pa
            "g": 9.81, # gravity constant, m/s^2
            "x_th_eps": 0.5, # backlash (should be as small as possible)
            "dx_th_eps": 0.1, # used as limit for positions checking
            "x_p_init": 1e3, # TODO: REWRITE GETTING THE INITIAL POSITION
        },
        **kwargs
    ):
        super().__init__(
            *args,
            system_parameters_init=system_parameters_init,
            **kwargs
        )
        """Droplet generator (hydraulic) system

        Args:
            system_parameters_init: parameters of the system:
                p_l_gauge: Gauge liquid pressure before throttle [Pa]. 
                    Defaults to 1.5e5.
                p_hydr_init_gauge: Gauge hydraulic container pressure [Pa]. 
                    Defaults to 0..
                x_th_limits: Real throttle position limits [µm]. 
                    Defaults to (0, 20).
                freq_th: Frottle frequency [Hz]. Defaults to 500.0.
                m_p: Piston mass [kg]. Defaults to 20e-3.
                D_th: Equivalent throttle diameter [m]. Defaults to 200e-6.
                D_hydr: Hydraulic container diameter [m]. Defaults to 20e-3.
                D_work: Working container diameter [m]. Defaults to 20e-3.
                D_exit: Exit orifice diameter [m]. Defaults to 0.33e-3.
                l_exit: Exit orifice length [m]. Defaults to 8.5e-3.
                p_c: Pressure difference on the piston to start movement [Pa].
                    Defaults to 10e3.
                eta: Mechanical efficiency. Defaults to 0.70.
                zeta_th: Hydraulic throttle coefficient. 
                    Might be find empirically (from real equipment). 
                    Now it is taken for the valve, see 
                    'Идельчик И. Е. Справочник по гидравлическим 
                    сопротивлениям. М., "Машиностроение", 1975'. 
                    Defaults to 5.0.
                rho_hydr: Hydraulic liquid density [kg/m^3]. Defaults to 1e3.
                rho_work: Working liquid density [kg/m^3]. Defaults to 1e3.
                --beta_v_hydr: Hydraulic liquid compressibility [Pa^-1]. 
                    Defaults to 0.49e-9.
                --beta_v_work: Working liquid compressibility [Pa^-1]. 
                    Defaults to 0.49e-9.
                sigma_work: Working liquid surface tension [N/m]. 
                    Defaults to 73e-3.
                mu_work: Working liquid viscosity[Pa*s]. Defaults to 1.0e-3.
                v_j: Jet speed for the stable operation (found experimentaly)
                    [mm/s]. Defaults to 200.
                --jet_length_std: Standard deviation of Relative jet length 
                    observation. Defaults to 5e-2.
                --jet_velocity_std: Standard deviation of Relative jet velocity 
                    observation. Defaults to 1e-2.
                p_atm: Atmosphere (ambient) pressure, Pa. 
                    Defaults to 1e5.
                g: Gravity constant, m/s^2. Defaults to 9.81. 
                x_th_eps: Backlash (should be as small as possible). 
                    Defaults to 0.5.
                dx_th_eps: Used as limit for positions checking.
                    Defaults to 0.1.
        """
        
        # Pressure and Throttle parameters
        p_atm, p_l_gauge, p_hydr_init_gauge = (
            self._parameters["p_atm"],
            self._parameters["p_l_gauge"],
            self._parameters["p_hydr_init_gauge"],
        )
        freq_th, x_th_limits = (
            self._parameters["freq_th"],
            self._parameters["x_th_limits"]
        )
        self.update_system_parameters(
            {
                "p_l": p_atm + p_l_gauge,
                "p_hydr_init": p_atm + p_hydr_init_gauge,
                # Max throttle speed, µm/s.
                "v_th_max": freq_th*(x_th_limits[1] - x_th_limits[0]),
            }
        )
        
        # Piston mass, gravity parameters; Gravity force
        m_p, g = (
            self._parameters["m_p"],
            self._parameters["g"]
        )
        self.update_system_parameters(
            {
                "F_g": m_p*g, # Gravity force, N
            }
        )

        # Geometry parameters        
        D_th, D_hydr, D_work, D_exit, l_exit = (
            self._parameters["D_th"],
            self._parameters["D_hydr"],
            self._parameters["D_work"],
            self._parameters["D_exit"],
            self._parameters["l_exit"],
        )
        # FUNCTION GET AREA BY DIAMETER
        # returns area in [m^2], if input in [m]
        get_area = lambda D: np.pi*D**2/4
        self.update_system_parameters(
            {
                "A_hydr": get_area(D_hydr), # m^2
                "A_work": get_area(D_work), # m^2
                "D_work_exit_2_ratio": D_work**2/D_exit**2,
            }
        )
        
        # Friction force
        A_hydr, A_work = self._parameters["A_hydr"], self._parameters["A_work"]
        A_max = max(A_hydr, A_work) # m^2
        # Coulomb friction force, N
        self.update_system_parameters(
            {
                "F_c": self._parameters["p_c"] * A_max
            }
        )
        # Exit hydraulic coefficient
        # see https://doi.org/10.1201/9781420040470
        C_D = 0.827 - 0.0085*l_exit/D_exit
        self.update_system_parameters(
            {
                "zeta_exit": 1/C_D**2
            }
        )
        
        # Liquid params
        rho_hydr, rho_work, sigma_work, mu_work = (
            self._parameters["rho_hydr"],
            self._parameters["rho_work"],
            self._parameters["sigma_work"],
            self._parameters["mu_work"],
        )
        
        # PRESSURE LOSSES
        zeta_th, zeta_exit = (
            self._parameters["zeta_th"], self._parameters["zeta_exit"]
        )
        self.update_system_parameters(
            {
                # Capillar pressure difference to othercome for drop exiting
                "p_capillar_max": 4*sigma_work/D_exit,
                "ploss_coef_hydr": (zeta_th*rho_hydr*D_hydr**4)/(32*D_th**2),
                "ploss_coef_work": (zeta_exit*rho_work*D_work**4)/\
                    (2e12*D_exit**4),
            }
        )
        
        # Dimensionless jet numbers
        v_j = self._parameters["v_j"]
        We_j = rho_work*v_j**2*D_exit/(1e6*sigma_work)
        Re_j = rho_work*v_j*D_exit/(1e3*mu_work)
        Oh_j = np.sqrt(We_j)/Re_j
         
        # JET LENGTH AND DROP DIAMETER
        # Critical jet length
        # see https://doi.org/10.1007/s00348-003-0629-6
        l_crit = 13.4e3*(np.sqrt(We_j)\
            + 3*We_j/Re_j) * D_exit
        # Estimated Droplet diameter
        D_drop = 1e3*(1.5*np.pi*np.sqrt(2 + 3*Oh_j))**(1/3) * D_exit
        self.update_system_parameters(
            {
                "l_crit": l_crit,
                "D_drop": D_drop,
            }
        )
        
        # # RESET piston position and last pressure in hydraulic container
        # self.reset()
        
        
    # def reset(self) -> None:
    #     """Reset system to initial state."""
    #     self.update_system_parameters(
    #         {
    #             # p_h|_{x_{th}>0}
    #             "_p_hydr_last": self._parameters["p_hydr_init"],
    #             # x_p|_{x_{th}>0}
    #             "_x_p_last": None, # define later, if None
    #             # Initial piston position
    #             "_x_p_init": None, # define later, if None
    #         }
    #     )


    def get_pressure_hydraulic(self, state) -> float:
        """ Get pressure in the hydraulic container

        Args:
            state: array of current state:
                x_p (float): piston position [µm]
                v_p (float): piston velocity [µm/s]
                x_th (float): throttle position [µm]

        Returns:
            float: pressure in the hydraulic container [Pa]
        """
        
        # State params
        x_p, v_p, x_th = state[0], state[1], state[2]
        
        # # Define last piston position first time as init piston position
        # if self._parameters["_x_p_last"] is None:
        #     self.update_system_parameters(
        #         {
        #             "_x_p_last": x_p,
        #         }
        #     )
        
        # System params
        p_l, x_th_eps, ploss_coef_hydr = ( # p_l, x_th_eps, ploss_coef_hydr, p_hydr_last, x_p_last, beta_v_hydr = (
            self._parameters["p_l"],
            self._parameters["x_th_eps"],
            self._parameters["ploss_coef_hydr"],
            # self._parameters["_p_hydr_last"],
            # self._parameters["_x_p_last"],
            # self._parameters["beta_v_hydr"],
        )

        # print('X_th: ', x_th)
        # print('x_th_eps: ', x_th_eps)

        pressure_hydraulic_open = (
            p_l
            - v_p*ploss_coef_hydr*rg.abs(v_p)/\
                (rg.if_else(x_th_eps > x_th, x_th_eps, x_th)**2)
        )
        # pressure_hydraulic_closed = (
        #     p_hydr_last + (x_p_last/x_p - 1)/beta_v_hydr
        # )

        pressure_hydraulic_open = rg.if_else(
            v_p != 0, pressure_hydraulic_open, p_l
        )
        pressure_hydraulic = pressure_hydraulic_open
        
        # pressure_hydraulic = rg.if_else(
        #     x_th > 0, pressure_hydraulic_open, pressure_hydraulic_closed
        # )
        # 
        # # DOES NOT WORK WITH CASADI
        # # Save piston position and hydraulic pressure if throttle is opened
        # if x_th > 0:
        #     self.update_system_parameters(
        #         {
        #             "_x_p_last": x_p,
        #             "_p_hydr_last": pressure_hydraulic,
        #         }
        #     )

        # print('pressure_hydraulic: ', pressure_hydraulic)

        return pressure_hydraulic
    
    
    def get_pressure_working(self, state) -> float:
        """ Get pressure in the working container

        Args:
            state: array of current state:
                x_p (float): piston position [µm]
                v_p (float): piston velocity [µm/s]
                x_th (float): throttle position [µm] (Do not required)

        Returns:
            float: pressure in the test container [Pa]
        """
        # State params
        x_p, v_p = state[0], state[1]

        # # Define init piston position
        # if self._parameters["_x_p_init"] is None:
        #     self.update_system_parameters(
        #         {
        #             "_x_p_init": x_p,
        #         }
        #     )
        
        # Required parameters
        p_atm, ploss_coef_work = ( # x_p_init, p_capillar_max, beta_v_work, p_atm, ploss_coef_work = (
            # self._parameters["_x_p_init"],
            # self._parameters["p_capillar_max"],
            # self._parameters["beta_v_work"],
            self._parameters["p_atm"],
            self._parameters["ploss_coef_work"],
        )
        
        # # Position difference
        # dx_p = x_p - x_p_init

        # p_compress = rg.abs(dx_p/x_p)/beta_v_work

        # p_capillar = rg.if_else(
        #     p_capillar_max < p_compress, p_capillar_max, p_compress
        # )

        pressure_working = p_atm # + rg.sign(dx_p) * p_capillar

        pressure_working = rg.if_else(
            v_p != 0,
            pressure_working + v_p*rg.abs(v_p)*ploss_coef_work,
            pressure_working
        )

        # print('pressure working: ', pressure_working)

        return pressure_working
    
    
    def get_force_hydraulic(self, state) -> float:
        """ Get hydraulic force acting on the piston

        Args:
            state: array of current state:
                x_p (float): piston position [µm] (Do not required)
                v_p (float): piston velocity [µm/s]
                x_th (float): throttle position [µm] (Do not required)

        Returns:
            float: hydraulic force [N]
        """
        
        p_hydr = self.get_pressure_hydraulic(state)
        p_work = self.get_pressure_working(state)
        
        A_hydr, A_work = self._parameters["A_hydr"], self._parameters["A_work"]
        # print("works: ", A_hydr, A_work)        
        return A_hydr*p_hydr - A_work*p_work
    
    
    def get_force_friction(self, state, F_h: float) -> float:
        """ Get friction force acting on the piston

        Args:
            state:
            v_p (float): array of current state:
                x_p (float): piston position [µm]. Do not required
                v_p (float): piston velocity [µm/s]
                x_th (float): throttle position [µm]. Do not required
            F_h (float): Hydraulic force [N]

        Returns:
            float: friction force [N]
        """
        
        v_p = state[1]
        
        # Required parameters
        F_c, eta, F_g = (
            self._parameters["F_c"],
            self._parameters["eta"],
            self._parameters["F_g"],
        )
        
        F_fr_h = (1-eta)*F_h
        
        F_fr_dynamic = -rg.sign(v_p) * rg.if_else(F_c > F_fr_h, F_c, F_fr_h)
        # If piston does not move
        F_fr_static = -rg.if_else(F_g + F_h > 0, 1, -1) * F_c #####################################
        
        F_fr = rg.if_else(v_p != 0, F_fr_dynamic, F_fr_static)

        # print('force freion: ', F_fr)

        return F_fr
    
    
    def get_acceleration(self, state) -> float:
        """ Get piston acceleration

        Args:
            state: array of current state:
                x_p (float): piston position [µm]
                v_p (float): piston velocity [µm/s]
                x_th (float): throttle position [µm]

        Returns:
            float: piston acceleration [m/s^2]
        """
        
        # State params
        v_p = state[1]
        
        F_h = self.get_force_hydraulic(state)
        F_fr = self.get_force_friction(state, F_h)
        
        # Required params
        F_g, g, m_p = (
            self._parameters["F_g"],
            self._parameters["g"],
            self._parameters["m_p"],
        )
        
        cond_velocity = rg.if_else(v_p != 0, 1, 0) ################################################################
        cond_fr_overcome = rg.if_else(rg.abs(F_h + F_g) > rg.abs(F_fr), 1, 0)
        
        # return 0, if piston does not move and acting force lower than friction
        # acceleration = rg.if_else(
            # (cond_velocity + cond_fr_overcome) > 0, # OR
        acceleration = 1e6*(g + 1/m_p * (F_h + F_fr))
            # 0.0001
        # )

        # print("accerelarton: ", acceleration)

        return acceleration
    
    
    def _compute_state_dynamics(self, time, state, inputs):
        
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )
        
        x_th_limits = self._parameters["x_th_limits"]

        x_th = rg.if_else(state[2] > x_th_limits[0], state[2], x_th_limits[0])
        x_th = rg.if_else(x_th < x_th_limits[1], x_th, x_th_limits[1])

        x_th_act = rg.if_else(
            inputs[0] > x_th_limits[0], inputs[0], x_th_limits[0]
        )
        x_th_act = rg.if_else(
            x_th_act < x_th_limits[1], x_th_act, x_th_limits[1]
        )
        
        # \dot{x_p}
        Dstate[0] = state[1]
        # \dot{v_p}
        Dstate[1] = self.get_acceleration(state)
        
        # TODO: make First-order aperiodic chain: 1/T * (k*x - y)
        dx_th_eps, v_th_max = (
            self._parameters["dx_th_eps"],
            self._parameters["v_th_max"]
        )
        # \dot{x_th}
        # if real throttle position is differ from the set one, change it
        # cond_th_dyn = (rg.abs(x_th_act - x_th) > dx_th_eps)
        # Dstate[2] = rg.if_else(
        #     cond_th_dyn, rg.sign(x_th_act - x_th) * v_th_max, 0
        # )
        # x_th = rg.if_else(cond_th_dyn, x_th, x_th_act)
        # state[2] = rg.if_else(cond_th_dyn, state[2], x_th)
        
        k = 1.
        Dstate[2] = v_th_max * (k*x_th_act - state[2])

        # print('Dstate: ', Dstate)

        return Dstate
    
    
    def get_clean_observation(self, state):
        """Get clean observations 
        (relative jet length and relative jet velocity), without sensors noise

        Args:
            state: system state

        Returns:
            observation (jet length, jet velocity)
        """
        x_p = state[0]
        v_p = state[1]
        
        observation = rg.zeros(
            self.dim_observation,
            prototype=state,
        )
        
        # print("State -1:", state)

        # # Define init piston position
        # if self._parameters["_x_p_init"] is None:
        #     self.update_system_parameters(
        #         {
        #             "_x_p_init": x_p,
        #         }
        #     )
        
        x_p_init, D_work_exit_2_ratio = (
            self._parameters["x_p_init"],
            self._parameters["D_work_exit_2_ratio"]
        )

        # print("paramters -1: ", x_p_init, D_work_exit_2_ratio)
        
        # Jet length
        observation[0] = (
            1e-3 * (x_p - x_p_init) * D_work_exit_2_ratio
        )
        
        # Jet velocity
        observation[1] = (
            1e-3 * v_p * D_work_exit_2_ratio
        )

        # print("Observationn -1: ", observation)
        
        return observation
    

    def _get_observation(self, time, state, inputs):
        """ Get observation with normal noise
        """
        observation = self.get_clean_observation(state)
        
        # # relative jet length with noise
        # observation[0] += np.random.normal(
        #     scale=self._parameters["jet_length_std"]
        # )
        # # relative jet velocity with noise
        # observation[1] += np.random.normal(
        #     scale=self._parameters["jet_velocity_std"]
        # )
        
        return observation