import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt

from pycmr_task import *
from pycmr_network import *
from ncms_analysis import *

# model parameters
param = Parameters('basic_tcm')
# could have named parameter sets stored in some imported file
setattr(param, 'beta_enc', 0.6)
setattr(param, 'beta_rec', 0.9)
setattr(param, 'P1', 3)
setattr(param, 'P2', 1)
setattr(param, 'L', 1)
setattr(param, 'Dfc', 3)
setattr(param, 'Dcf', 1)
setattr(param, 'T', 0.35)
setattr(param, 'stop_fn', 'exponential')
setattr(param, 'X1', 0.005)
setattr(param, 'X2', 0.7)
#setattr(param, 'stop_fn', 'fixed')
#setattr(param, 'X1', 0.1)

# task stores information about the immediate free recall task
task = Task('ifr_task')
setattr(task, 'n_trials', 120)
#setattr(task, 'n_trials', 1)
setattr(task, 'list_length', 24)
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
setattr(fixed, 'beta_enc', 0.6)
#setattr(fixed, 'beta_rec', 0.9)
setattr(fixed, 'P1', 3)
setattr(fixed, 'P2', 1)
setattr(fixed, 'L', 1)
setattr(fixed, 'Dfc', 3)
setattr(fixed, 'Dcf', 1)
setattr(fixed, 'T', 0.35)
setattr(fixed, 'stop_fn', 'exponential')
setattr(fixed, 'X1', 0.005)
#setattr(fixed, 'X2', 0.7)

free_names = ('beta_rec', 'X2')
#free_names = ('beta_rec')
bounds = [(0, 1), (0, 2)]
#bounds = [(0, 1)]


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

param_vec = np.array([0.9, 0.7])
#param_vec = np.array([0.8])

# there's a bug in this code, if the list of names of free parameters only has one name in it, \
# it is treated as a string rather than a tuple inside the eval function, so free_names[i] just grabs 
# the first character of the name for setattr instead of the whole name.  I haven't figured out how 
# the differential_evolution code works well enough yet, have to check if the same issue occurs when 
# the fn gets called from within differential_evolution.  One solution could be to have a more elaborate 
# param structure that gets passed in, instead of a list of strings

LLorig = eval_param_ifr(param_vec, ftuple[0], ftuple[1], ftuple[2], ftuple[3])
#print('altered: {:.2f}; unaltered: {:.2f}'.format(LL1,LL2))

print('LL with generating params: {:.2f}'.format(LLorig))


# result = differential_evolution(eval_param_ifr, bounds, args=(ftuple), maxiter=3)
# result = differential_evolution(eval_param_ifr, bounds, args=(ftuple), maxiter=3)

# print('DE solution: {}; LL: {}'.format(result.x, result.fun))
print('done')