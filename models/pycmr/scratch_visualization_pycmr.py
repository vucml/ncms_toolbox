

import numpy as np
import matplotlib.pyplot as plt
from ncms_classes import *


# set params, make a net, set up simulation

# model parameters

param = Parameters('basic_tcm')
setattr(param, 'beta_enc', 0.7)
setattr(param, 'P1', 3)
setattr(param, 'P2', 1)
setattr(param, 'L', 1)
setattr(param, 'Dfc', 1)
setattr(param, 'Dcf', 1)
setattr(param, 'T', 1)

task = Task('ifr_task')
setattr(task, 'list_length', 20)
setattr(task, 'pres_itemnos', np.array(range(task.list_length)))
setattr(task, 'units_needed', task.list_length)


# create a network
# I guess most of this would go in an init network function
net = Network('cmr_basic')
net.initialize_basic_tcm(param, task.units_needed)

print('\n -+^ I did not crash ^+- \n')

# after study list, plot to demonstrate the end-of-list contextual state

#fig = figure()
#plt.plot()







