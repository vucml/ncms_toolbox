import numpy as np
from numpy.linalg import norm


class ExemplarModel(object):
    """The basic exemplar model of memory as originated by Hintzman (1984, 1986, 1988) in MINERVA 2.

    In the model, every `experience` is represented as a vector - an ordered list of feature values along
    many dimensions. A record of each experience - called a `trace` is stored as a new, separate row in a m x n
    `memory` matrix where rows correspond to memory traces and columns correspond to feature dimensions.

    To retrieve information from memory, a feature vector can be presented as a `probe`. The probe activates all traces
    in memory in parallel. Each trace's `activation` is a cubed function of its `similarity` to the probe. The sum of
    these traces weighted by their activation represents an `echo` summarizing the memory system's response to the
    probe. The content and intensity of this echo can serve downstream behavior such as recognition or word sense
    disambiguation or (hopefully) free recall.

    Attributes:
        memory: a m x n array where rows correspond to accumulated memory traces and columns correspond to feature dims
    """

    def __init__(self, experiences=None):
        """Inits exemplar model with initial set of experiences in memory (if any)."""
        self.memory = None
        if experiences:
            self.experience(experiences)

    def experience(self, experiences):
        """Adds new experience(s) to model memory, represented as new row(s) in the model's memory array."""
        self.memory = np.vstack((self.memory, np.array(experiences))) if self.memory else np.array(experiences)

    def probe(self, probe):
        """Presents a cue to memory system, fetching an echo reflecting its pattern of activation across traces.

        The probe activates all traces in memory in parallel. Each trace's `activation` is a cubed function of its
        `similarity` to the probe. The sum of these traces weighted by their activation is an `echo` summarizing
        the memory system's response to the probe.
        """

        # computes and cubes similarity value to find activation for each trace in memory
        activation = np.power(np.sum(self.memory * probe, axis=1) / (norm(self.memory) * norm(probe)), 3)

        # multiply each trace by its associated activation and take a column-wise sum to retrieve echo
        echo = np.sum((self.memory.T * activation).T, axis=0)
        return echo

    def compare_probes(self, first_probe, second_probe):
        """Compute the resemblance (cosine similarity) between the echoes associated with probes A and B."""
        echoes = self.probe(first_probe), self.probe(second_probe)
        return np.sum(echoes[0] * echoes[1]) / (norm(echoes[0]) * norm(echoes[1]))
