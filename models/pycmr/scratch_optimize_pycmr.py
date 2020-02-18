import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt

from pycmr_task import *
from pycmr_network import *
from ncms_analysis import *

# model parameters
param = Parameters('basic_tcm')
# could have named parameter sets stored in some imported file
setattr(param, 'beta_enc', 0.8)
setattr(param, 'beta_rec', 0.8)
setattr(param, 'P1', 100)
setattr(param, 'P2', 10)
setattr(param, 'L', 1)
setattr(param, 'Dfc', 10)
setattr(param, 'Dcf', 1)
setattr(param, 'T', 2)
setattr(param, 'stop_fn', 'exponential')
setattr(param, 'X1', 0.005)
setattr(param, 'X2', 0.7)
#setattr(param, 'stop_fn', 'fixed')
#setattr(param, 'X1', 0.1)

# task stores information about the immediate free recall task
task = Task('ifr_task')
setattr(task, 'n_trials', 500)
#setattr(task, 'n_trials', 1)
setattr(task, 'list_length', 15)
setattr(task, 'pres_itemnos', np.array(range(task.list_length)))
# specific to the unit vector style of item representations
setattr(task, 'units_needed', task.list_length+1)
setattr(task, 'serial_position', 1)

recalls = task.ifr_task_generate(param)

from scipy.optimize import differential_evolution

# now can attempt to recover generating parameters
# using ifr_task_predict
# need to create a wrapper function, working backwards

fixed = Parameters('fixed params')
#setattr(param, 'beta_enc', 0.8)
setattr(fixed, 'beta_rec', 0.8)
setattr(fixed, 'P1', 100)
setattr(fixed, 'P2', 10)
setattr(fixed, 'L', 1)
#setattr(fixed, 'Dfc', 10)
setattr(fixed, 'Dcf', 1)
setattr(fixed, 'T', 2)
setattr(fixed, 'stop_fn', 'exponential')
setattr(fixed, 'X1', 0.005)
setattr(fixed, 'X2', 0.7)

free_names = ('beta_enc', 'Dfc')
bounds = [(0, 1), (0, 100)]



def eval_param_ifr(x, recalls, param, free_names, task):
    # args[0] is recalls matrix
    # args[1] is fixed parameters
    # args[2] is names of free parameters
    # args[3] is task structure
    #
    # add current values of free params onto fixed param structure
    # recalls = args[0]
    # param = args[1]
    for i in range(len(x)):
        setattr(param, free_names[i], x[i])
    # task = args[3]
    #
    LL = task.ifr_task_predict(recalls, param)
    return -LL

# args is a tuple of any additional fixed parameters
ftuple = (recalls, fixed, free_names, task)

# test out the eval param function
#param_vec = np.array([0.5, 8])
#LL1 = eval_param_ifr(param_vec, ftuple)
param_vec = np.array([0.8, 10])
LLorig = eval_param_ifr(param_vec, ftuple[0], ftuple[1], ftuple[2], ftuple[3])
#print('altered: {:.2f}; unaltered: {:.2f}'.format(LL1,LL2))

print('LL with generating params: {:.2f}'.format(LLorig))


# result = differential_evolution(eval_param_ifr, bounds, args=(ftuple), maxiter=3)
result = differential_evolution(eval_param_ifr, bounds, args=(ftuple), maxiter=3)

print('DE solution: {}; LL: {}'.format(result.x, result.fun))
