# %% [markdown]
# # Homonym Experiment
# Reproduction of some results from **Experiment 1: Homonyms in an Artificial Language** in Jamieson et al (2018).
#
# ## Artificial Language Corpus
# In the experiment, `words` are represented as a unique vector where each dimension takes a randomly sampled value
# from a normal distribution with mean zero and variance 1/n. Experiences are encoded as the sum of the word vectors
# occurring in a given context.
#
# A simple artificial language is constructed to generate a corpus of experiences (specified in Table 1 of paper),
# consisting of 12 words sorted between 7 lexical categories and 3 sentence frames grammatically specifying how
# triplets of words from different categories can be associated within a verbal context.
#
# To explore whether an exemplar model can predict human judgements even in the case of homonyms - words with the same
# spelling/pronunciation but different meanings - 20,000 grammatical sentences are sampled from the artificial language
# and encoded as memory traces within an ExemplarModel instance.
#
# ## Analyses
# We replicate two analyses (just one for now):
# 1. We confirm that words that occur in similar contexts have similar meanings by comparing their echoes against
#       their co-occurrence frequencies.
# TODO: 2. Second, we reproduce disambiguation of the meaning of a homonym depending on the context in which it is
#       presented by comparing model responses to various sentences instead of individual words.
#       
# Experiment results are saved in `/results`.
# %%
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from models.ExemplarModel import ExemplarModel


def artificial_language_corpus():
    """Generate a corpus of experiences to serve examination of ExemplarModel

    `words` are represented as a unique vector where each dimension takes a randomly sampled value
    from a normal distribution with mean zero and variance 1/n. Experiences are encoded as the sum of the word vectors
    occurring in a given context.

    A simple artificial language is constructed to generate a corpus of experiences (specified in Table 1 of paper),
    consisting of 12 words sorted between 7 lexical categories and 3 sentence frames grammatically specifying how
    triplets of words from different categories can be associated within a verbal context.

    To explore whether an exemplar model can predict human judgements even in the case of homonyms - words with the same
    spelling/pronunciation but different meanings - 20,000 grammatical sentences are sampled from the artificial
    language and encoded as memory traces within an ExemplarModel instance.
    """

    # random vectors for each word
    word_list = ['man', 'woman', 'car', 'truck', 'plate', 'glass', 'story', 'news', 'stop', 'smash', 'report', 'break']
    word_vectors = {word: np.random.normal(0, np.sqrt(1 / 20000), 20000) for word in word_list}

    # categories of lexical items
    lexical_items = {
        'NOUN_HUMAN': ['man', 'woman'],
        'NOUN_VEHICLE': ['car', 'truck'],
        'NOUN_DINNERWARE': ['plate', 'glass'],
        'NOUN_NEWS': ['story', 'news'],
        'VERB_VEHICLE': ['stop', 'break'],
        'VERB_DINNERWARE': ['smash', 'break'],
        'VERB_NEWS': ['report', 'break'],
    }

    # sample sentences and experiences
    sentences, experiences = [], []
    frames = np.random.choice([1, 2, 3], 20000)
    for i in range(20000):
        if frames[i] == 1:
            sentence = [np.random.choice(lexical_items['NOUN_HUMAN']),
                        np.random.choice(lexical_items['VERB_VEHICLE']),
                        np.random.choice(lexical_items['NOUN_VEHICLE'])]
        elif frames[i] == 2:
            sentence = [np.random.choice(lexical_items['NOUN_HUMAN']),
                        np.random.choice(lexical_items['VERB_DINNERWARE']),
                        np.random.choice(lexical_items['NOUN_DINNERWARE'])]
        else:
            sentence = [np.random.choice(lexical_items['NOUN_HUMAN']),
                        np.random.choice(lexical_items['VERB_NEWS']),
                        np.random.choice(lexical_items['NOUN_NEWS'])]

        sentences.append(sentence)
        experiences.append(np.sum([word_vectors[word] for word in sentence], axis=0))

    return {'word_list': word_list, 'word_vectors': word_vectors, 'lexical_items': lexical_items,
            'sentences': sentences, 'experiences': experiences}


# %%
def contextual_similarity_experiment():
    """
    Generates similarity matrix visualization testing whether words occurring in similar contexts have similar echoes.

    We compute and visualize a pairwise similarity matrix comparing echoes associated with each unique word to one 
    another. Items in the same lexical categories (e.g. 'story' and 'news') occur in similar contexts and so should 
    have similar echoes. Items in opposing lexical categories (e.g. 'story' and 'smash') should be found dissimilar. 
    Items that occur equally often in every context ('break') should be somewhere in the middle.
    """

    # initiate model with corpus of word contexts as experiences
    corpus = artificial_language_corpus()
    model = ExemplarModel(corpus['experiences'])

    # compute pairwise similarities for each word in list
    similarities = np.full((len(corpus['word_list']), len(corpus['word_list'])), np.nan)
    for x in tqdm(range(len(corpus['word_list']))):
        for y in range(len(corpus['word_list'])):
            if x == y:
                continue
            word, other_word = corpus['word_list'][x], corpus['word_list'][y]
            similarities[x, y] = model.compare_probes(corpus['word_vectors'][word], corpus['word_vectors'][other_word])

    sns.heatmap(similarities, xticklabels=corpus['word_list'], yticklabels=corpus['word_list'],
                annot=True, linewidths=.5, cmap="YlGnBu")
    plt.show()
    return corpus, similarities


# %%
def main():
    return contextual_similarity_experiment()


if __name__ == '__main__':
    result = main()

# %% [markdown]
# ## Conclusion
# As shown, the monogamous words from the vehicle topic are clustered together
# (i.e., stop, car, truck), the monogamous words from the dinnerware topic are
# clustered together (i.e., plate, glass, smash), the monogamous words from
# the news topic are clustered together (i.e., story, news, report),
# and the promiscuous words (i.e., words that occurred in all three topics) are
# clustered together (i.e., man, woman, and break).
