import pytest
import numpy as np

from regelum.scenario import Scenario
from regelum.simulator import SciPy
from regelum.simulator import CasADi

import sys
sys.path.append('./')

from src.system import HydraulicSystemSimpleRg
from src.policy import PDController

from regelum.utils import RCType
from regelum.utils import rg

from regelum import callback

sampling_time = 1e-3

system = callback.detach(HydraulicSystemSimpleRg)()

simulator = CasADi( #SciPy(
    system=system,
    state_init = rg.array([1e3, 0, 0], rc_type=RCType.CASADI),
    time_final = 10e-3,
)

scenario = Scenario(
    policy=PDController(
        system=system,
        sampling_time=sampling_time
    ),
    simulator=simulator,
    sampling_time=0.01,
    N_episodes=1,
    N_iterations=1,
)

scenario.run()