
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt

from pycmr_task import *
from pycmr_network import *
from ncms_analysis import *

# model parameters
param = Parameters('basic_tcm')
# could have named parameter sets stored in some imported file
setattr(param, 'beta_enc', 0.5)
setattr(param, 'beta_rec', 0.5)
setattr(param, 'P1', 3)
setattr(param, 'P2', 1)
setattr(param, 'L', 1)
setattr(param, 'Dfc', 3)
setattr(param, 'Dcf', 1)
# setattr(param, 'sampling_fn', 'classic')
setattr(param, 'T', 0.35)
setattr(param, 'stop_fn', 'exponential')
setattr(param, 'X1', 0.001)
setattr(param, 'X2', 0.6)

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

B_rec_vals = np.linspace(0, 1, 11)
LL=np.zeros(B_rec_vals.shape)
for i in range(len(B_rec_vals)):
    setattr(param,'beta_rec',B_rec_vals[i])
    LL[i] = task.ifr_task_predict(recalls, param)

quick_plot_spc(quick_spc(recalls, task.list_length))
plt.plot(B_rec_vals,LL,'ko-')
plt.show()
print('hi')