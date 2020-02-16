import numpy as np
import numpy.random as rn

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
        self.recalled_items = []
        
        # while loop over recall events 
        stopped = False
        while not stopped:
            # prompt a recall
            this_event = net.recall_attempt_basic_tcm(param, self)
            self.recall_attempt += 1
            results.append(this_event[0])
            # check if it was a stop event
            if this_event==self.list_length:
                stopped = True
            else:
                # reactivate the winner
                net.reactivate_item_basic_tcm(this_event, param, self)
                
        return results


# allows item to act as a data struct
class Item:

    def __init__(self,name):
        self.name = name
        self.string = ''
        self.word_pool_id = []
        # index relates item to network when using unit vectors
        self.index = []

