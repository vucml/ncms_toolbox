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
setattr(task, 'list_length', 20)
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

results = task.ifr_trial_generate(net,param)

# check the network, grab activation state of c
context_state = net.c_layer.act_state
print(results)

print('\n -+^ I did not crash ^+- \n')

# after study list, plot to demonstrate the end-of-list contextual state

fig = plt.figure(1)
plt.plot(range(task.list_length+1), context_state, 'b.')
plt.xlabel('context element index')
plt.ylabel('activation state')
plt.title('context state immediately after list presentation')
plt.show()






