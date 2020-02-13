import numpy as np


class Environment:
    
    def __init__(self, name):
        self.name = name
        self.network = []
        self.task = []

class Task:
    
    def __init__(self, name):
        self.name = name
        self.stimulus_pool = []

class Network:

    def __init__(self,name):
        self.name = name
        self.layer_list = []
        self.projection_list = []

    def add_layer(self, layer):
        self.layer_list.append(layer)

    def add_projection(self, proj):
        self.projection_list.append(proj)
    
    def initialize_layers_zeros(self):
        for this_layer in self.layer_list:
            # maybe only if verbose?
            print('initializing ' + this_layer.name)
            this_layer.initialize_zeros()

class Layer:
    
    def __init__(self, name, n_units):
        self.name = name
        self.n_units = n_units
        self.net_input = np.zeros( (self.n_units,1) )
        self.act_state = np.zeros( (self.n_units,1) )

    def initialize_zeros(self):
        self.net_input = np.zeros( (this_layer.n_units,1) )
        self.act_state = np.zeros( (this_layer.n_units,1) )

    def activate_unit_vector(self, index):
        self.act_state[index] = 1.0

class Projection:
    
    def __init__(self, name, from_layer, to_layer):
        self.name = name    
        self.from_layer = from_layer
        self.to_layer = to_layer
        # rows index units in to_layer, columns index units in from_layer
        # given that projection operation will be matrix * from_layer.act_state
        self.matrix = np.zeros( (self.to_layer.n_units, self.from_layer.n_units) )
        # if false, projection will not project activity during simulation?
        # self.active = True

    def init_matrix_identity(self):
        if self.from_layer.n_units==self.to_layer.n_units:
            self.matrix = np.eye(self.from_layer.n_units)
        else:
            print('Problem in init_matrix_identity, from and to layers need to have same number of units.')

    def project_activity(self):
        # act_state of from_layer projects along matrix to influence net_input of to_layer
        # net_input is potentially aggregating from multiple incoming connections, so
        # it may be necessary to initialize it to zeros elsewhere
        temp_incoming = np.dot(self.matrix, self.from_layer.act_state)
        self.to_layer.net_input = self.to_layer.net_input + temp_incoming

## create a network
#net = Network('cmr_basic')
## create f and c layers
#f_layer = Layer('f', 10)
#c_layer = Layer('c', 10)
## associate them with the network
#net.add_layer(f_layer)
#net.add_layer(c_layer)
## create projections and associate them with the network
#m_fc_pre = Projection('m_fc_pre', f_layer, c_layer)
#m_fc_exp = Projection('m_fc_exp', f_layer, c_layer)
#net.add_projection(m_fc_pre)
#net.add_projection(m_fc_exp)
## pre projection uses an identity matrix, exp is initialized as zeros
#m_fc_pre.init_matrix_identity()

## activate an item on the f layer / activate a simple unit vector representation on f
## and project it to c (it updates the net input of c)
#index = 0
#f_layer.activate_unit_vector(index)
#m_fc_pre.project_activity()
## update activation state of c, which triggers integration operation


#print('hi')
