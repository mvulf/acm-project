import numpy as np

from regelum.policy import Policy
from regelum.utils import rg

# Import for MPC
from regelum.model import Model, ModelNN
from regelum.optimizable.core.configs import OptimizerConfig

from regelum.system import ComposedSystem, System
from numpy import ndarray
from regelum.predictor import EulerPredictor
from regelum.objective import RunningObjective
from regelum.data_buffers import DataBuffer
from regelum.objective import mpc_objective


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


class MPC(Policy):

    def __init__(
        self,
        model: Model | ModelNN,
        system: System | ComposedSystem,
        action_bounds: list | ndarray | None,
        optimizer_config: OptimizerConfig | None,
        prediction_horizon: int,
        running_objective: RunningObjective,
        predictor: EulerPredictor,
        discount_factor: float | None = 1,
        epsilon_random_parameter: float | None = None,
    ):
        """Instantiate MPC policy."""
        super().__init__(
            model,
            system,
            action_bounds,
            optimizer_config,
            discount_factor,
            epsilon_random_parameter,
        )  # Initialize the parent `Policy` class.

        self.prediction_horizon = prediction_horizon
        self.running_objective = running_objective
        self.predictor = predictor
        ###### Define the optimization problem
        self.observation_var = self.create_variable(
            1,  # dimensionality of axis 0
            self.system.dim_observation,  # dimensionality of axis 1
            name="observation",
            is_constant=True,  # is_constant set to `True` as `observation` is a constant parameter of optimization
        )
        self.est_state_var = self.create_variable(
            1,  # dimensionality of axis 0
            self.system.dim_state,  # dimensionality of axis 1
            name="estimated_state",
            is_constant=True,  # is_constant is set to `True` as `estimated_state` is a constant parameter of optimization
        )
        self.policy_model_weights_var = self.create_variable(
            name="policy_model_weights",
            is_constant=False,  # is_constant is set to False because policy_model_weights is a decision variable in our optimization problem
            like=self.model.named_parameters,  # like parameter utilizes the dimensions of the model's weights for compatibility
        )
        ## Let us register bounds for policy model weights to be within action bounds
        (
            self.action_bounds_tiled,
            self.action_initial_guess,
            self.action_min,
            self.action_max,
        ) = self.handle_bounds(
            self.action_bounds,
            self.dim_action,
            tile_parameter=self.model.weights.shape[0],
        )
        self.register_bounds(self.policy_model_weights_var, self.action_bounds_tiled)

        ## Make `Optimizable` aware of objective function and variables it depends on
        self.register_objective(
            self.cost,
            variables=[
                self.observation_var,
                self.est_state_var,
                self.policy_model_weights_var,
            ],
        )

    def optimize(self, databuffer: DataBuffer) -> None:
        """Define optimization routine for `Optimizable` class."""
        new_weights = super().optimize_symbolic(
            **databuffer.get_optimization_kwargs(
                keys=["observation", "estimated_state"],
                optimizer_config=self.optimizer_config,
            ),
            policy_model_weights=self.policy_model_weights_var(),
        )[
            "policy_model_weights"
        ]  # Get the optimized weights from `Optimizable` class
        self.model.update_and_cache_weights(new_weights)

    def cost(self, observation, estimated_state, policy_model_weights):
        """Cost function for MPC Policy."""
        return mpc_objective(
            observation=observation,
            estimated_state=estimated_state,
            policy_model_weights=policy_model_weights,
            discount_factor=self.discount_factor,
            running_objective=self.running_objective,
            prediction_horizon=self.prediction_horizon,
            predictor=self.predictor,
            model=self.model,
        )  # Call `mpc_objective` function to get the cost of current state and sequence of predicted actions