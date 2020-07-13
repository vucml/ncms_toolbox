# %% [markdown]
# # Basic Simulations
# A series of simulations to explore the functionality of instance_cmr.
#
# # Model
# %%
import numpy as np
from numpy.linalg import norm

class InstanceCMR(object):
    """The context maintenance and retrieval model re-imagined as an exemplar model.

    As typical of exemplar models, every `experience` is represented as a high-dimensional feature vector.
    A record of each experience - called a `trace` - is stored as a new, separate row in a m x n `memory` matrix
    where rows correspond to memory traces and columns correspond to feature dimensions.

    As in a retrieved context model, a contextual representation is also maintained. Compared to the representation of
    current experience, it changes slowly over time, reflecting a recency-weighted average of information related to
    recently presented stimuli. New memory traces associate studied items with the context active during presentation.
    This enables context-driven recall of items, and item-driven recall of context.

    To retrieve information from memory, a feature vector can be presented as a `probe`. The probe activates all traces
    in memory in parallel. Each trace's `activation` is a cubed function of its `similarity` to the probe. The sum of
    these traces weighted by their activation represents an `echo` summarizing the memory system's response to the
    probe. The content and intensity of this echo is the information that characterizes memory performance across tasks.

    Attributes:
        memory: a m x n array where rows correspond to accumulated memory traces and columns correspond to feature dims
        context: length-n vector reflecting a recency-weighted average of recently presented stimuli information
        item_count: number of unique items in experiment list identifying the relevant store of pre-experimental memory
        drift_rate: rate of context drift during item processing
        shared_support: uniform amount of support items initially have for one another in recall competition
        learning_rate: controls contribution of experimental associations relative to pre-experimental associations
        stop_probability_scale: scaling of the stop probability over output position
        stop_probability_growth: rate of increase in stop probability
        choice_sensitivity: sensitivity parameter of the Luce choice rule
    """

    def __init__(self, item_count, drift_rate, shared_support, learning_rate,
                 stop_probability_scale, stop_probability_growth, choice_sensitivity):
        """Starts exemplar model with initial set of experiences in memory.

        For the prototype, we assume items are orthonormal in their features and use unique index vectors to represent
        them as such. To represent pre-experimental memory, a trace is initially laid for each item representing its
        vector representation modified to have some parametrized amount of shared_support for all items (similar to
        CMR's alpha) and another parameter experiment_weight (similar to CMR's gamma) controlling the contribution of
        experimental memory relative to pre-experimental memory to echo representations.

        Args:
            item_count: number of unique experimental items identifying the relevant store of pre-experimental memory
            drift_rate: rate of context drift during item processing
            shared_support: uniform amount of support items initially have for one another in recall competition
            learning_rate: controls contribution of experimental memory relative to pre-experimental memory
            stop_probability_scale: scaling of the stop probability over output position
            stop_probability_growth: rate of increase in stop probability
        """
        # store initial parameters
        self.item_count = item_count
        self.drift_rate = drift_rate
        self.shared_support = shared_support
        self.learning_rate = learning_rate
        self.stop_probability_scale = stop_probability_scale
        self.stop_probability_growth = stop_probability_growth
        self.choice_sensitivity = choice_sensitivity

        # initialize memory and context
        self.context = np.zeros((item_count,))
        self.memory = np.eye(item_count)
        self.memory[np.logical_not(np.eye(item_count, dtype=bool))] = self.shared_support

    def experience(self, experiences):
        """Adds new trace(s) to model memory, represented as new row(s) in the model's memory array. The stored
        experience is the context representation after it's been updated by the current experience.
        """
        if len(np.shape(experiences)) == 1:
            experiences = [experiences]
        for experience in experiences:
            self.update_context(np.array(experience))
            self.memory = np.vstack((self.memory, self.context))

    def update_context(self, experience):
        """Updates contextual representation based on content of current experience."""

        # retrieves echo (memory information) associated w/ experience to serve as input to context
        # parallel operation to equation 10 from Morton & Polyn (2016)
        context_input = self.probe(experience)
        context_input = context_input / norm(context_input)  # normalized to have length 1

        # updated context is sum of current context and input modulated to have len 1 w/ rho and a specified drift_rate
        # parallel operation to equations 11-12 from Morton & Polyn (2016)
        rho = np.sqrt(1 + (np.power(self.drift_rate, 2) * (np.power(self.context * context_input, 2) - 1))) - (
                self.drift_rate * (self.context * context_input))
        self.context = (rho * self.context) + (self.drift_rate * context_input)

    def probe(self, probe):
        """Presents a cue to memory system, fetching an echo reflecting its pattern of activation across traces.

        The probe activates all traces in memory in parallel. Each trace's `activation` is a cubed function of its
        `similarity` to the probe. The sum of these traces weighted by their activation is an `echo` summarizing
        the memory system's response to the probe. The learning_rate parameter further weights the relative contribution
        of pre-experimental and experimental traces to activity patterns.
        """
        activation = np.power(np.sum(self.memory * probe, axis=1) / (norm(self.memory) * norm(probe)), 3)
        activation *= np.hstack((np.ones((self.item_count,)) * 1 - self.learning_rate,
                                 np.ones((len(self.memory) - self.item_count,)) * self.learning_rate))
        echo = np.sum((self.memory.T * activation).T, axis=0)
        return echo

    def compare_probes(self, first_probe, second_probe):
        """Compute the resemblance (cosine similarity) between the echoes associated with probes A and B."""
        echoes = self.probe(first_probe), self.probe(second_probe)
        return np.sum(echoes[0] * echoes[1]) / (norm(echoes[0]) * norm(echoes[1]))

    def free_recall(self):
        """Simulates performance on a free recall task based on experienced items.

        We initialize context similar to eq. 16 from Morton & Polyn (2016), simulating end-of-list distraction and
        some amount of pre-list context reinstatement. This context is used as a retrieval cue (probe) to attempt
        retrieval of a studied item, generating an associated memory echo.

        At each recall attempt, we also calculate a probability of stopping recall as a function of output position
        according to eq. 18 of Morton & Polyn (2016). The probability of recalling a given item conditioned on not
        stopping recall is defined on the basis of the item's similarity to the current contextual representation
        according to a formula similar to eq 19 of Morton and Polyn (2016).
        """

        # drift context toward the pre-experimental context then perform recall until stop is triggered
        recall, items, preretrieval_context = [], np.eye(self.item_count), self.context
        self.update_context(np.zeros((self.item_count,)))
        while True:

            # compute outcome probabilities and make choice based on distribution
            outcome_probabilities = np.zeros((self.item_count + 1))
            outcome_probabilities[0] = self.stop_probability_scale * np.exp(len(recall) * self.stop_probability_growth)
            for j in range(len(outcome_probabilities)):
                outcome_probabilities[j + 1] = np.power(
                    np.sum(self.context, items[j]) / (norm(self.context) * norm(items[j])), self.choice_sensitivity)
            outcome_probabilities[1:] *= (1 - outcome_probabilities[0]) / np.sum(outcome_probabilities[1:])
            choice = np.random.choice(len(outcome_probabilities), p=outcome_probabilities)

            # store and resolve outcome
            if not choice:
                break
            recall.append(choice - 1)
            self.update_context(self.probe(items[choice - 1]))

        self.context = preretrieval_context
        return recall


# %% [markdown]
# ## Setup
# Using an arbitrary set of parameters selected from `ncms_toolbox/models/cymr/tutorial.py`, we generate a
# couple dozen simulations of a recall experiment with 6 trials and 24 items.

# %%
#import os
#os.chdir('..')

import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
#from models.InstanceCMR import InstanceCMR

param = {'item_count': 24, 'drift_rate': .5, 'shared_support': 0, 'learning_rate': .5,
         'stop_probability_scale': .001, 'stop_probability_growth': .5, 'choice_sensitivity': 1}

# %% [markdown]
# ## Are Memory Traces All Unit Vectors?
# To find out, we generate a model w/ some sample experiences and compute the magnitude of each trace.
#
# %%
model = InstanceCMR(**param)
model.experience(np.eye(param['item_count']))
model.memory

# %%
np.linalg.norm(model.memory, axis=1)

# %%
