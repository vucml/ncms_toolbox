import numpy as np
from ncms_classes import *

# create an environment 
env = Environment('ifr_session')

# create generic stimulus pool
# generic: items are unit vectors
stim_pool = Stimulus_Pool('generic_items')
# this is the number of items in the pool as opposed to on a list
n_items = 20
stim_pool.construct_items('unit_vectors', n_items)
# create stimulus property
# when item is presented, iterate through stim properties
# in matlab code, env keeps track of which units are earmarked for stimulus features vs distraction features
# if unit vectors are being used, don't need to store a matrix of item representations, just an item index

# create a task object
task = Task('ifr_task')
# 'ifr', 'dfr', 'cdfr'?
# task.set_task_type('ifr')
# add task to environment
env.add_task(task)
# start by doing it here then move it into the classes

# trial object? 
# need pres_itemnos 
setattr(task, 'list_length', 20)
setattr(task, 'pres_itemnos', np.array(range(task.list_length)))
setattr(task, 'units_needed', task.list_length)

# model parameters
# still have to figure out how parameters are dealt with 
param = Parameters('basic_tcm')
setattr(param, 'beta_enc', 0.7)
setattr(param, 'P1', 3)
setattr(param, 'P2', 1)
setattr(param, 'L', 1)
setattr(param, 'Dfc', 1)
setattr(param, 'Dcf', 1)
setattr(param, 'T', 1)

# create a network
# I guess most of this would go in an init network function
net = Network('cmr_basic')
# add network to environment
env.add_network(net)


# initialization function could check environment to see how many units needed? 
# is n_units a parameter? not clearly
net.initialize_basic_tcm(param, task.units_needed)

# STUDY PERIOD

# sample study period, broken down into pieces
# doing this without a stim pool first, then can consider role of stim pool

# activate start unit, maybe this is a special operation
index = 0
init_beta = 1
# here is broken! have to figure out right way to structure the layers and projections
# can this use setattr?
net.f_layer.activate_unit_vector(index)
net.m_fc_pre.project_activity()
net.c_layer.integrate_net_input(init_beta)

# activate an item on the f layer / activate a simple unit vector representation on f
# and project it to c (it updates the net input of c)

# if you are presenting a list, presumably this would be a task function?
# functional unit needs to be present_item
for i in range(task.list_length):
    env.serial_position = i+1
    item = task
    net.present_item_basic_tcm(param, env, item)

# RECALL PERIOD
# env keeps track of output positions

# loop over output positions
# code differs if this is generative vs predictive
# predictive code iterates over recall events
# generative code, while loop until stop condition is triggered

stopped = False
while not stopped: 
    # simplest version is project c to f_in, sampling function, prob choice rule
    # clear out f net_input
    net.f_layer.initialize_net_input_zeros()
    # project from c to f net input
    net.m_cf_pre.project_activity()
    net.m_cf_exp.project_activity()
    # sampling function modifies net input? or is it applied in transformation to act state?
    # act state should probably just be for reactivated item, act state is what actually gets projected back up to c
    # if the different sampling functions are actually different functions instead of options for one fn, is that easier? 
    # in the matlab code the sampling fn creates a strength vector separate from f_in
    net.f_layer.sampling_fn_classic(param.T)
    # if net_input is all zeros, matlab code sets equiv non-zero support for every item 
    # strength vector is a stage on the way to constructing a vector of probabilities
    
    # calculate stop probability
    stop_prob = 0.1
    # calculate probabilities for individual items
    # need the strength values for the recallable items
    prob_vec = np.zeros(())
    
    stopped = True

print('hi')
