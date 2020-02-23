

import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt

from pycmr_task import *
from pycmr_network import *
from ncms_analysis import *

which_param = 'mat_demo'
## These parameters are pulled from the KragEtal matlab demo in SVN ncms_toolbox 

if which_param == 'mat_demo':
    # model parameters
    param = Parameters('KragEtal15_base')
    # could have named parameter sets stored in some imported file
    setattr(param, 'beta_enc', 0.3257)
    setattr(param, 'beta_rec', 0.8554)
    setattr(param, 'P1', 1.5699)
    setattr(param, 'P2', 0.4177)
    setattr(param, 'L', 1)
    setattr(param, 'C', 0.0503) 
    setattr(param, 'B_ri', 0.8162)
    setattr(param, 'B_ipi', 0.8935)
    setattr(param, 'B_s', 0.2402)
    setattr(param, 'T', 1)
    # equiv of G = 0.2321
    G = 0.2321
    Dfc = (1-G)/G
    setattr(param, 'Dfc', Dfc)
    setattr(param, 'Dcf', 0)
    setattr(param, 'stop_fn', 'exponential')
    setattr(param, 'X1', 0.001)
    setattr(param, 'X2', 2.2656)
    setattr(param, 'sampling_rule', 'power')

#elif which_param == 'scratch_demo':
    




