import numpy as np
import numpy.random as rn

    
class Environment:
    
    def __init__(self, name):
        self.name = name
        self.network = []
        self.task = []
        self.serial_position = []

    def add_network(self, net):
        self.network = net

    def add_task(self, task):
        self.task = task

# consider whether stim pool needs a class
# maybe: generic stim pool assumes network is initialized after each trial?
# this would mean you only need as many items as max list length?
# stim pool as class: has an associated network, or network has associated stim pool? 
# or both stim pool and network are associated with environment
# have to specify, when an item is presented from stim pool, which layer does it activate?
# to make CMR09, where potentially both item and source features activate on separate layers, 
# need to have option for multiple stimulus properties I guess
# if stim pool is an object, can have range of item numbers, a type property 'unit_vectors'?
class Stimulus_Pool:
    
    def __init__(self, name):
        self.name = name

    # have to think about this, how to do this flexibly, seems like you need to know how many units are in the target layer
    def construct_items(self, item_type, n_items): 
        self.item_number = np.zeros((n_items))
        # check item type
        if item_type == 'unit_vectors':
            self.item_type = item_type
            # possible to do this without for loop?
            for i in range(n_items):
                self.item_number[i] = i

    # if you have unit vectors, each item can have an index attribute
    # which indicates which unit to activate when the item is presented
    def link_pool_to_network(self, net, layer):
        if self.item_type == 'unit_vectors':
            print("here")



