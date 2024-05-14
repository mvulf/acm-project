from regelum.simulator import Simulator
import scipy as sp


class SciPy(Simulator):
    """Class for SciPy integrators."""

    class SciPySolver:
        """Class to wrap Integratod into a uniform API. 
        """
        
        def __init__(
            self,
            simulator: "SciPy",
            rtol:float=1e-3,
            atol:float=1e-6,
        ):
            # TODO: Edit docstring
            """Initialize a SciPySolver object.

            Args:
                time_final (float): The final time for the solver.
                step_size (float): The step size for the solver.
                action_init (np.array): The initial action for the
                    solver.
                system (System): The system object for the solver.
            """
            self.simulator = simulator
            # self.integrator = simulator.integrator
            self.time_final = simulator.time_final
            self.step_size = simulator.max_step
            self.time = 0.0
            self.system = simulator.system
            self.rtol = rtol
            self.atol = atol
            
        def step(self):
            """Conduct one simulation step with action which set in the system
            """
            
            if self.time >= self.time_final:
                raise RuntimeError("An attempt to step with a finished solver")
            
            numerical_results = sp.integrate.solve_ivp(
                fun=self.system.compute_closed_loop_rhs,
                t_span=(self.time, self.time + self.step_size),
                y0=self.simulator.state,
                rtol=self.rtol,
                atol=self.atol,
            )
            self.time = numerical_results.t[-1]
            self.state = numerical_results.y.T[-1] # TODO: Check dimensionality
        
        @property
        def t(self):
            return self.time

        @property
        def y(self):
            return self.state
            
            
    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        self._state = new_state


    def initialize_ode_solver(self):
        assert self.time_final is not None and self.max_step is not None, (
            "Must specify time_final and max_step"
            + " in order to initialize SciPy solver"
        )
        # INSTEAD OF (sp.integrate.RK45) GET OBJECT OF CLASS which is BASED ON Simulator from DropControl
        # with the method **step()**
        # self.time = self.ODE_solver.t - LAST TIME 
        # self.state = self.ODE_solver.y - LAST STATE
        # ODE_solver = sp.integrate.RK45(
        #     self.system.compute_closed_loop_rhs,
        #     self.time, # initial time
        #     self.state, # initial state
        #     self.time_final, # t_bound
        #     max_step=self.max_step,
        #     first_step=self.first_step,
        #     atol=self.atol,
        #     rtol=self.rtol,
        # )
        
        ODE_solver = self.SciPySolver(self)
        return ODE_solver
    
    
    def reset(self):
        if self.system.system_type == "diff_eqn":
            self.time = 0.0
            self.ODE_solver = self.initialize_ode_solver()
            self.state = self.state_init
            self.observation = self.get_observation(
                time=self.time, state=self.state_init, inputs=self.action_init
            )
        else:
            self.time = 0.0
            self.observation = self.get_observation(
                time=self.time, state=self.state_init, inputs=self.system.inputs
            )
    