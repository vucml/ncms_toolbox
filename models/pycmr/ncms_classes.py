import numpy as np
import numpy.random as rn

class Parameters:
    
    def __init__(self, name):
        self.name = name

    # def set_parameter(self, param_name, param_val):

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

class Task:
    
    def __init__(self, name):
        self.name = name
        self.stimulus_pool = []
        self.item_list = []

    def ifr_trial_generate(self, net, param):
        # alternative/companion is ifr_trial_predict
        # assumes net is initialized

        # task function for the study list presentation code
        # because sometimes will want to run study list without recall period
        # for loop over list items / list length
        self.ifr_study_list(net, param)

        # task function for basic free recall period
        results = self.ifr_recall_period(net, param)
        
        return results

    def ifr_study_list(self, net, param):
        #
        self.serial_position = 1

        # initialize context by presenting 'start_item'
        # using index = list_length + 1 and beta = 1
        net.initialize_context(self)
                
        for i in range(self.list_length):
            # make a generic item for now
            self.item_list.append(Item('generic'))
            self.item_list[-1].index = i
            # task keeps track of serial position
            net.present_item_basic_tcm(param, self)
            self.serial_position += 1

    def ifr_recall_period(self, net, param):
        #
        results = []
        self.recall_attempt = 1
        
        # while loop over recall events 
        stopped = False
        while not stopped:
            # prompt a recall
            this_event = net.recall_attempt_basic_tcm(param, self)
            results.append(this_event[0])
            # check if it was a stop event
            if this_event==self.list_length:
                stopped = True
            
        return results


# allows item to act as a data struct
class Item:

    def __init__(self,name):
        self.name = name
        self.string = ''
        self.word_pool_id = []
        # index relates item to network when using unit vectors
        self.index = []


class Network:

    def __init__(self,name):
        self.name = name
        self.layer_list = []
        self.projection_list = []

    def add_layer(self, layer_name, layer):
        setattr(self,layer_name,layer)
        #self.layer_list.append(layer)

    def add_projection(self, proj_name, proj):
        setattr(self, proj_name, proj)
        #self.projection_list.append(proj)
    
    # consider whether this fn should be removed
    def initialize_layers_zeros(self):
        for this_layer in self.layer_list:
            # maybe only if verbose?
            print('initializing ' + this_layer.name)
            this_layer.initialize_zeros()

    def initialize_basic_tcm(self, param, n_units):
        # create f and c layers
        f_layer = Layer('f', n_units)
        c_layer = Layer('c', n_units)        
        # associate them with the network
        self.add_layer('f_layer', f_layer)
        self.add_layer('c_layer', c_layer)
        # create projections and associate them with the network
        m_fc_pre = Projection('m_fc_pre', f_layer, c_layer)
        m_fc_exp = Projection('m_fc_exp', f_layer, c_layer)
        m_cf_pre = Projection('m_cf_pre', c_layer, f_layer)
        m_cf_exp = Projection('m_cf_exp', c_layer, f_layer)
        self.add_projection('m_fc_pre', m_fc_pre)
        self.add_projection('m_fc_exp', m_fc_exp)
        self.add_projection('m_cf_pre', m_cf_pre)
        self.add_projection('m_cf_exp', m_cf_exp)
        # pre projection uses an identity matrix, exp is initialized as zeros
        # D parameters, should be able to scale the identity matrix by a parameter
        m_fc_pre.init_matrix_identity(param.Dfc)
        m_cf_pre.init_matrix_identity(param.Dcf)
        # m_fc_exp has associated learning rule?        

    def present_item_basic_tcm(self, param, task):
        # get ready for the next simulated item by zeroing out net_input of c
        self.c_layer.initialize_net_input_zeros()
        # when unit vector representations used, index tells which f unit to activate
        index = task.item_list[-1].index
        self.f_layer.activate_unit_vector(index)
        # project activity uses projection to update net_input of to_layer
        self.m_fc_pre.project_activity()
        # update activation state of c, which triggers integration operation
        # should integration rate be a stored property of the layer, or just of the integration function?
        self.c_layer.integrate_net_input(param.beta_enc)
        # Hebbian learning on m_fc_exp 
        # for full cmr implementation, primacy scaling must be possible
        # will prob need environment to track serial position
        primacy_scaling = (param.P1 * np.exp(-1 * param.P2 * (task.serial_position-1))) + 1
        # primacy scaling modifies learning rate but just in the cf direction
        self.m_fc_exp.hebbian_learning(param.L)
        self.m_cf_exp.hebbian_learning(param.L+primacy_scaling)        

    def initialize_context(self, task):
        self.c_layer.initialize_net_input_zeros()
        index = task.list_length
        self.f_layer.activate_unit_vector(index)
        self.m_fc_pre.project_activity()
        # integrate fully, beta = 1
        self.c_layer.integrate_net_input(1)

    def recall_attempt_basic_tcm(self, param, task):
        # print('implement recall attempt basic tcm')
        # task can keep track of previous recalls

        # uniform sampling to get things going
        # outcomes [0, LL) represent study items being recalled
        # outcome LL represents recall termination

        # calculate stop probability
        # fixed stop prob to get things going
        # can check param to see what the stop rule is
        
        outcomes = task.list_length+1
        prob_vec = np.ones(outcomes)
        prob_vec = prob_vec / np.sum(prob_vec)
        this_event = rn.choice(outcomes, 1, p=prob_vec)
        print(this_event)
        

        return this_event
        #this_rnd = rn.rand()
        #print('random number: {0}'.format(this_rnd))
        #return this_rnd
        
class Layer:
    
    def __init__(self, name, n_units):
        self.name = name
        self.n_units = n_units
        self.net_input = np.zeros( (self.n_units) )
        self.act_state = np.zeros( (self.n_units) )
        # integration rate 1 means no integration
        self.integration_rate = 1;

    def initialize_net_input_zeros(self):
        self.net_input = np.zeros( (self.n_units) )
        
    def normalize_net_input(self):
        self.net_input = self.net_input / np.sqrt(np.sum(self.net_input**2))

    def initialize_act_state_zeros(self):
        self.act_state = np.zeros( (self.n_units) )

    def activate_unit_vector(self, index):
        self.initialize_act_state_zeros()
        self.act_state[index] = 1.0

    def integrate_net_input(self, beta):
        # does this require normalizing net_input vector first?
        self.normalize_net_input()
        incoming_dot = np.dot(self.act_state, self.net_input)
        rho = np.sqrt(1 + beta**2 * (incoming_dot**2 - 1)) - (beta * incoming_dot)
        self.act_state = (rho * self.act_state) + (beta * self.net_input)

    def sampling_fn_classic(self,T):
        self.net_input = np.exp((2*self.net_input) / T)
        
    def update_activation(self):
        pass

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

    def init_matrix_identity(self, scaling_factor):
        if self.from_layer.n_units==self.to_layer.n_units:
            self.matrix = np.eye(self.from_layer.n_units) * scaling_factor
        else:
            print('Problem in init_matrix_identity, from and to layers need to have same number of units.')

    def project_activity(self):
        # act_state of from_layer projects along matrix to influence net_input of to_layer
        # net_input is potentially aggregating from multiple incoming connections, so
        # it may be necessary to initialize it to zeros elsewhere
        temp_incoming = np.dot(self.matrix, self.from_layer.act_state)
        self.to_layer.net_input = self.to_layer.net_input + temp_incoming

    def hebbian_learning(self, learning_rate):
        # the optimized code just updates the needed row or col when unit vectors are used 
        # this will be slower; could have a parallel set of optimized functions?
        # hebbian outer product 
        self.matrix = self.matrix + np.outer(self.to_layer.act_state,self.from_layer.act_state) * learning_rate
