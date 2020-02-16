
import numpy as np
import matplotlib.pyplot as plt
from ncms_classes import *

# some helper functions

# make a serial position curve
def quick_spc(recalls, list_length):
    # assumes clean recalls: no repeats or intrusions
    n_trials = recalls.shape[0]
    numer = np.zeros(list_length)
    for i in range(list_length):
        # serial positions start at 1
        this_spos = i + 1
        # how many times does this spos appear
        numer[i] = np.sum(recalls==this_spos)
    spos_prec = numer / n_trials
    return spos_prec

def quick_plot_spc(spos_prec):
    fig = plt.figure()
    plt.plot(range(len(spos_prec)),spos_prec,'ko-')
    plt.show()

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

task = Task('ifr_task')
# task stores information about the immediate free recall task
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

