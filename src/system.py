
from regelum.system import System


class HydraulicSystem(System):
    
    _name = 'HydraulicSystem'
    _system_type = 'diff_eqn'
    _dim_state = 5
    _dim_inputs = 1
    _dim_observation = 2
    _state_naming = [
        "piston position [µm]", 
        "piston velocity [µm/s]", 
        "throttle position [µm]",
        "hydraulic pressure [Pa]",
        "working pressure [Pa]",
    ]
    _observation_naming = [
        "jet length [mm]", 
        "jet velocity [mm/s]",
    ]
    _inputs_naming = ["throttle_action"]
    _action_bounds = [[-20.0, 20.0]]
    
    def __init__(
        self,
        *args,
        system_parameters_init = {
            "p_l_gauge": 1.5e5,
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
        },
        **kwargs,
    ):
        """Droplet generator (hydraulic) system

        Args:
            system_parameters_init: parameters of the system:
                p_l_gauge: Gauge liquid pressure before throttle [Pa]. 
                    Defaults to 1.5e5.
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
        """
        
        super().__init__(
            *args,
            system_parameters_init=system_parameters_init,
            **kwargs
        )
        
        # Get absolute pressure
        p_atm, p_l_gauge = (
            self._parameters["p_atm"],
            self._parameters["p_l_gauge"],
        )
        self.update_system_parameters(
            {
                "p_l": p_atm + p_l_gauge,
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
        
        # TODO: Calculate discharge coefficient
        