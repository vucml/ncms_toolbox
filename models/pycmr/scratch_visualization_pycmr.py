import numpy as np
import matplotlib.pyplot as plt
from ncms_classes import *

# set params, make a net, set up simulation

# model parameters
param = Parameters('basic_tcm')
# could have named parameter sets stored in some imported file
setattr(param, 'beta_enc', 0.7)
setattr(param, 'P1', 3)
setattr(param, 'P2', 1)
setattr(param, 'L', 1)
setattr(param, 'Dfc', 1)
setattr(param, 'Dcf', 1)
setattr(param, 'T', 1)

task = Task('ifr_task')
# task stores information about the immediate free recall task
setattr(task, 'n_trials', 15)
setattr(task, 'list_length', 10)
setattr(task, 'pres_itemnos', np.array(range(task.list_length)))
# specific to the unit vector style of item representations
setattr(task, 'units_needed', task.list_length+1)
setattr(task, 'serial_position', 1)

# create a network
net = Network('cmr_basic')
net.initialize_basic_tcm(param, task.units_needed)

# results, a list of integers
# outcomes [0, LL) represent study items being recalled
# outcome LL represents recall termination
recalls = np.zeros((task.n_trials, task.list_length), dtype=int)

for i in range(task.n_trials):
    results = task.ifr_trial_generate(net,param)
    these = np.array(results[:-1])
    # print(type(these))
    these = these + 1
    # add +1 if we want recalls matrix to be in terms of serial position
    recalls[i,:len(these)] = these
    
# data structure
# recalls matrix, n trials by n recall events


# check the network, grab activation state of c
context_state = net.c_layer.act_state
# print(results)
print(recalls)


print('\n -+^ I did not crash ^+- \n')

# after study list, plot to demonstrate the end-of-list contextual state

fig = plt.figure(1)
plt.plot(range(task.list_length+1), context_state, 'b.')
plt.xlabel('context element index')
plt.ylabel('activation state')
plt.title('context state immediately after list presentation')
#plt.show()






