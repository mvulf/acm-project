import pytest
import numpy as np

from regelum.scenario import Scenario
from regelum.simulator import SciPy

from src.system import HydraulicSystem
from src.policy import PDController

sampling_time = 1e-3

system = HydraulicSystem()

simulator = SciPy(
    system=system,
    state_init = np.array([1e3, 0]),
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