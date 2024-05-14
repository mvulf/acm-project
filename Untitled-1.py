# %%
import pytest
import numpy as np

from regelum.scenario import Scenario
from regelum.simulator import SciPy
from regelum.simulator import CasADi


import sys
sys.path.append('../../')


from regelum.system import KinematicPoint
from regelum.utils import RCType
from regelum.utils import rg

from regelum import callback

sampling_time = 1e-3

system = callback.detach(KinematicPoint)()

simulator = CasADi(
    system=system,
    state_init = rg.array([1e3, 0, 0], rc_type=RCType.CASADI),
    time_final = 10e-3,
)
