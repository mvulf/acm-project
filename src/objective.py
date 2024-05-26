import torch
from regelum.model import ModelQuadLin
from regelum.utils import rg

from src.system import StationaryHydraulicSystem

class HydraulicObjectiveModel(ModelQuadLin):
    """This objective model substracts target (l_crit) from current
    observations and then calculates quadratic form
    """
    def __init__(
        self,
        system: StationaryHydraulicSystem,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.l_crit = system._parameters["l_crit"]
    
    def forward(self, inputs, weights=None):
        if weights is None:
            weights = self.weights
            quad_matrix = self._quad_matrix
            linear_coefs = self._linear_coefs
            if isinstance(inputs, torch.Tensor):
                quad_matrix = torch.FloatTensor(quad_matrix).to(inputs.device)
                if linear_coefs is not None:
                    linear_coefs = torch.FloatTensor(linear_coefs).to(inputs.device)
        else:
            quad_matrix, linear_coefs = self.get_quad_lin(weights)

        # Substract target: l_crit
        substract = rg.zeros(
            inputs.shape,
            prototype=inputs
        )
        # Set target a little bit larger than critical length
        substract[:,0] = 1.005*self.l_crit
        inputs -= substract

        return ModelQuadLin.quadratic_linear_form(
            inputs,
            quad_matrix,
            linear_coefs,
        )