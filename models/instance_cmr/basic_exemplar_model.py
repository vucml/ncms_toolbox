# %% [markdown]
# # Basic Exemplar Model
# Here we attempt to reproduce and test a version of the MINERVA 2 exemplar model as originated by Hintzman (1984, 1986, 1988) and extended by Jamieson et al (2018) to model semantic memory.
#
# ## Overview
# In the model, every `experience` is represented as a vector - an ordered list of feature values along many dimensions. A record of each experience - called a `trace` is stored as a new, separate row in a m x n `memory` matrix where rows correspond to memory traces and columns correspond to feature dimensions.
#
# To retrieve information from memory, a feature vector can be presented as a `probe`. The probe activates all traces in memory in parallel. Each trace's `activation` is a cubed function of its `similarity` to the probe. The sum of these traces weighted by their activation represents an `echo` summarizing the memory system's response to the probe. The content and intensity of this echo can serve downstream behavior such as recognition or word sense disambiguation or (hopefully) free recall.
#
# ## Implementation

# %%
import numpy as np
from numpy.linalg import norm
from functools import reduce

class MINERVA_2(object):
    
    def __init__(self, experiences=None):
        self.memory = np.array(experiences)
        assert len(np.shape(self.memory)) <= 2, 'experiences not encodable as array w/ dim <= 2!'

    def experience(self, experiences):
        self.memory = np.vstack((self.memory, np.array(experiences)))
    
    def probe(self, probe):
        if len(np.shape(probe)) < 2:
            activation = 1-np.power(np.dot(self.memory, probe)/(norm(self.memory)*norm(probe)), 3)
        else:
            activation = reduce(np.multiply,
                    [1-np.power(np.dot(self.memory, p)/(norm(self.memory)*norm(p)), 3) for p in p])
        echo = np.average(self.memory, axis=0, weights=activation)
        return echo
    
    def compare(self, A, B):
        echoes = self.probe(A), self.probe(B)
        return 1- np.dot(echoes[0], echoes[1])/(norm(echoes[0])*norm(echoes[1]))


# %% [markdown]
# ## Experiment
# We reproduce the information in Figure 2 from Jamieson et al (2018) to confirm that the model is implemented correctly.
#
# In the experiment, `words` are represented as a unique vector where each dimension takes a randomly sampled value from a normal distribution with mean zero and variance 1/n. Experiences are encoded as the sum of the word vectors occurring in a given context.
#
# A simple artifical language is constructed to generate a corpus experiences (Table 1 of paper), consisting of 12 words sorted between 7 lexical categories and 3 sentence frames specifying how trios of words from different categories can be associated within a verbal context.
#
# The experiment works as follows:
# 1. We generate a random vector of dimensionality 20,000 for each word in the language. 
# 2. We sample 20,000 sentences from the artificial language.
# 3. We encode a representation of each sentence as a trace in memory.
# 4. We retrieve an echo for each of the invidiual words.
# 5. We compute the similiarty between the echo retrieved for each word against the echo retrieved for each of the other 11 words.
# 6. We'll also retrieve the echo for an ambiguous word (`break`) in conjunction with other words and compare the resulting echoes with the echoes for unambiguous words to confirm that the model can achieve word disambiguation.
#
# ### Experiences

# %%
# random vectors for each word
word_list = ['man', 'woman', 'car', 'truck', 'plate', 'break',
             'glass', 'story', 'news', 'stop', 'smash', 'report']
word_vectors = {word: np.random.normal(0, np.sqrt(1/20000), 20000) for word in word_list}

# categories of lexical items
NOUN_HUMAN = 'man', 'woman'
NOUN_VEHICLE = 'car', 'truck'
NOUN_DINNERWARE = 'plate', 'glass'
NOUN_NEWS = 'story', 'news'
VERB_VEHICLE = 'stop', 'break'
VERB_DINNERWARE = 'smash', 'break'
VERB_NEWS = 'report', 'break'

# sample sentences and experiences
sentences, experiences = [], []
frames = np.random.choice([1, 2, 3], 20000)
for i in range(20000):
    if frames[i] == 1:
        sentence = [np.random.choice(NOUN_HUMAN),
                    np.random.choice(VERB_VEHICLE),
                    np.random.choice(NOUN_VEHICLE)]
    elif frames[i] == 2:
        sentence = [np.random.choice(NOUN_HUMAN),
                    np.random.choice(VERB_DINNERWARE),
                    np.random.choice(NOUN_DINNERWARE)]
    else:
        sentence = [np.random.choice(NOUN_HUMAN),
                    np.random.choice(VERB_NEWS),
                    np.random.choice(NOUN_NEWS)]
    sentences.append(sentence)
    experiences.append(np.sum([word_vectors[word] for word in sentence], axis=0))

# %% [markdown]
# ### Semantic Similarity

# %%
import seaborn as sns
from tqdm import tqdm

model = MINERVA_2(experiences)
similarities = []
for probe_B in tqdm(word_list):
    similarities.append([model.compare(word_vectors[probe_A], word_vectors[probe_B]) for probe_A in word_list])

# %%
sns.heatmap(similarities, xticklabels=word_list, yticklabels=word_list)

# %%
similarities

# %% [markdown]
# The similarity metric seems wrong.

# %%
