from regelum.simulator import Simulator

class SciPy(Simulator):
    """Class for SciPy integrators."""

    def initialize_ode_solver(self):
        import scipy as sp

        ODE_solver = sp.integrate.RK45(
            self.system._compute_state_dynamics,
            0, # initial time
            self.state, # initial state
            self.time_final, # t_bound
            max_step=self.max_step,
            first_step=self.first_step,
            atol=self.atol,
            rtol=self.rtol,
        )
        return ODE_solver
    
    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        self._state = new_state