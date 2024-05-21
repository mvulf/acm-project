from regelum.simulator import Simulator
import casadi

from regelum.utils import rg

class CasADi(Simulator):
    """Class for CasADi integrators."""

    class CasADiSolver:
        """Nested class to wrap casadi integrator into a uniform API."""

        def __init__(
            self,
            simulator: "CasADi",
        ):
            """Initialize a CasADiSolver object.

            Args:
                integrator (casadi.integrator): A CasADi integrator
                    object.
                time_final (float): The final time for the solver.
                step_size (float): The step size for the solver.
                action_init (np.array): The initial action for the
                    solver.
                system (System): The system object for the solver.
            """
            self.simulator = simulator
            self.integrator = simulator.integrator
            self.time_final = simulator.time_final
            self.step_size = simulator.max_step
            self.time = 0.0
            self.system = simulator.system

        def step(self):
            """Advance the solver by one step."""
            if self.time >= self.time_final:
                raise RuntimeError("An attempt to step with a finished solver")
            state_new = (
                self.integrator(x0=self.simulator.state, p=self.system.inputs)["xf"]
                .full()
                .T
            )

            self.time += self.step_size
            self.state = state_new

        @property
        def t(self):
            return self.time

        @property
        def y(self):
            return self.state

    def initialize_ode_solver(self):
        self.integrator = self.create_CasADi_integrator(self.max_step)
        assert self.time_final is not None and self.max_step is not None, (
            "Must specify time_final and max_step"
            + " in order to initialize CasADi solver"
        )
        ODE_solver = self.CasADiSolver(self)
        return ODE_solver

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        self._state = new_state
        if hasattr(self, "integrator"):
            self.integrator.state = self.state

    def create_CasADi_integrator(self, max_step):
        state_symbolic = rg.array_symb(self.system.dim_state, literal="x")
        action_symbolic = rg.array_symb(self.system.dim_inputs, literal="u")
        time = rg.array_symb((1, 1), literal="t")

        ODE = self.system.compute_state_dynamics(
            time, state_symbolic, action_symbolic, _native_dim=True
        )
        DAE = {"x": state_symbolic, "p": action_symbolic, "ode": ODE}

        # options = {"tf": max_step, "abstol": self.atol, "reltol": self.rtol}
        # options = {"abstol": self.atol, "reltol": self.rtol}
        # options = {"tf": max_step}
        integrator = casadi.integrator("intg", "rk", DAE, 0, max_step) #tf = max_step, options

        # integrator = casadi.integrator("intg", "rk", DAE, 0, max_step)

        return integrator