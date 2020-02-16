import numpy as np
import matplotlib.pyplot as plt

from ncms_task import *
from ncms_model import *
from ncms_analysis import *
#from ncms_classes import *

# set params, make a net, set up simulation

# model parameters
param = Parameters('basic_tcm')
# could have named parameter sets stored in some imported file
setattr(param, 'beta_enc', 0.8)
setattr(param, 'beta_rec', 0.8)
setattr(param, 'P1', 20)
setattr(param, 'P2', 3)
setattr(param, 'L', 1)
setattr(param, 'Dfc', 10)
setattr(param, 'Dcf', 1)
setattr(param, 'T', 2)
setattr(param, 'stop_fn', 'exponential')
setattr(param, 'X1', 0.005)
setattr(param, 'X2', 0.3)
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


# generate n trials recall sequences
recalls = np.zeros((task.n_trials, task.list_length), dtype=int)

for i in range(task.n_trials):
    # create a network
    net = Network('cmr_basic')
    net.initialize_basic_tcm(param, task.units_needed)
    results = task.ifr_trial_generate(net, param)
    these = np.array(results[:-1])
    # print(type(these))
    these = these + 1
    # add +1 as we want recalls matrix to be in terms of serial position
    recalls[i,:len(these)] = these

#print(recalls)
quick_plot_spc(quick_spc(recalls, task.list_length))

