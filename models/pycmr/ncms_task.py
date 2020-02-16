import numpy as np
import numpy.random as rn
from ncms_model import *

class Task:
    
    def __init__(self, name):
        self.name = name
        self.stimulus_pool = []
        self.item_list = []

    def ifr_task_generate(self, param):
        # generate n trials recall sequences
        # using the same matrix format as EMBAM behavioral toolbox
        recalls = np.zeros((self.n_trials, self.list_length), dtype=int)
        # should it also generate other aspects of the data struct
        # like pres_itemnos?

        # generate n_trials worth of recall data
        for i in range(self.n_trials):
            # create and initialize network
            net = Network('cmr_basic')
            net.initialize_basic_tcm(param, self.units_needed)
            results = self.ifr_trial_generate(net, param)
            these = np.array(results[:-1])
            # add +1 as we want recalls matrix to be in terms of serial position
            these = these + 1
            recalls[i,:len(these)] = these
        return recalls
        
    def ifr_task_predict(self, recalls, param):
        # get predictive likelihood for synthetic data
        LL = 0.
        for i in range(recalls.shape[0]):
            # create and initialize network
            net = Network('cmr_basic')
            net.initialize_basic_tcm(param, self.units_needed)
            # results could have fields for overall likelihood and indiv item likelihood
            # or could just return indiv event likelihood
            # QUESTION: should a fn like this take recalls in this form?
            likelihood = self.ifr_trial_predict(recalls[i,:], net, param)
            LL += sum(likelihood)
        return LL

    def ifr_trial_generate(self, net, param):
        # alternative/companion is ifr_trial_predict
        # assumes net is initialized

        # task function for the study list presentation code
        self.ifr_study_list(net, param)

        # task function for basic free recall period
        results = self.ifr_recall_period_generate(net, param)
        
        return results

    def ifr_trial_predict(self, recalls, net, param):
        # convert recalls vector into indices with stop code at the end (LL)
        temp = recalls[recalls>0]
        rec_events = np.append(temp,[self.list_length])
        
        # assumes net is initialized
        # task function for the study list presentation code
        # this part is identical to the generate version
        self.ifr_study_list(net, param)

        # task function for basic free recall period
        likelihood = self.ifr_recall_period_predict(rec_events, net, param)
        
        return likelihood

    
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

    def ifr_recall_period_generate(self, net, param):
        #
        results = []
        self.recall_attempt = 1
        self.recalled_items = []
        
        # while loop to generate recall events 
        stopped = False
        while not stopped:
            # get probabilities of possible events
            prob_vec = net.prob_recall_basic_tcm(param, self)
            # prompt generation of recall evetn
            this_event = rn.choice(self.list_length+1, 1, p=prob_vec)
            self.recalled_items.append(this_event)
            self.recall_attempt += 1
            # log the event
            results.append(this_event[0])
            # check if it was a stop event
            if this_event==self.list_length:
                stopped = True
            else:
                # reactivate the winner
                net.reactivate_item_basic_tcm(this_event, param, self)
                
        return results

    def ifr_recall_period_predict(self, rec_events, net, param):
        likelihood = np.zeros(rec_events.shape)
        self.recall_attempt = 1
        self.recalled_items = []

        # for loop over provided recall events
        for i in range(len(rec_events)):
            # get probabilities of possible events
            prob_vec = net.prob_recall_basic_tcm(param, self)
            # grab the prob of observed event, transform to log
            likelihood[i] = np.log(prob_vec[rec_events[i]])
            # if it isn't the stop event, reactivate the winner
            if rec_events[i] != self.list_length:
                net.reactivate_item_basic_tcm(rec_events[i], param, self)
        return likelihood
                
# allows item to act as a data struct
class Item:

    def __init__(self,name):
        self.name = name
        self.string = ''
        self.word_pool_id = []
        # index relates item to network when using unit vectors
        self.index = []

