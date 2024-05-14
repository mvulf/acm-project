from regelum.simulator import SciPy
import scipy as sp

class SciPy(SciPy):
    """Updated class for SciPy integrators."""

    def initialize_ode_solver(self):
        
        # INSTEAD OF (sp.integrate.RK45) GET OBJECT OF CLASS which is BASED ON Simulator from DropControl
        # with the method **step()**
        # self.time = self.ODE_solver.t - LAST TIME 
        # self.state = self.ODE_solver.y - LAST STATE
        ODE_solver = sp.integrate.RK45(
            self.system.compute_closed_loop_rhs,
            self.time, # initial time
            self.state, # initial state
            self.time_final, # t_bound
            max_step=self.max_step,
            first_step=self.first_step,
            atol=self.atol,
            rtol=self.rtol,
        )
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
    