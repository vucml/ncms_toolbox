
import numpy as np
import pandas as pd
from cymr import fit
from cymr import cmr
from cymr import network
from cymr import parameters
from cymr.fit import Recall

class TestRecall(Recall):

    def prepare_sim(self, data, study_keys=None, recall_keys=None):
        data_study = data.loc[data['trial_type'] == 'study']
        data_recall = data.loc[data['trial_type'] == 'recall']
        merged = fr.merge_lists(data_study, data_recall)
        study = fr.split_lists(merged, 'study', ['input'])
        recalls = fr.split_lists(merged, 'recall', ['input'])
        return study, recalls

    def likelihood_subject(self, study, recalls, param_def, weights=None,
                           patterns=None):
        p = 2 - (param_def.fixed['x'] + 2) ** 2
        eps = 0.0001
        if p < eps:
            p = eps
        n = 1
        return np.log(p), n

    def generate_subject(self, study_dict, recall_dict, param_def, patterns=None, weights=None, **kwargs):
        recalls_list = [param_def.fixed['recalls']]
        # data = fit.add_recalls(study, recalls_list)
        return recalls_list

def data():
    data = pd.DataFrame(
        {'subject': [1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2],
         'list': [1, 1, 1, 1, 1, 1,
                  2, 2, 2, 2, 2, 2],
         'trial_type': ['study', 'study', 'study',
                        'recall', 'recall', 'recall',
                        'study', 'study', 'study',
                        'recall', 'recall', 'recall'],
         'position': [1, 2, 3, 1, 2, 3,
                      1, 2, 3, 1, 2, 3],
         'item': ['absence', 'hollow', 'pupil',
                  'hollow', 'pupil', 'empty',
                  'fountain', 'piano', 'pillow',
                  'pillow', 'fountain', 'pillow'],
         'item_index': [0, 1, 2, 1, 2, np.nan,
                        3, 4, 5, 5, 3, 5],
         'task': [1, 2, 1, 2, 1, np.nan,
                  1, 2, 1, 1, 1, 1]})
    return data

def data2():
    data = pd.DataFrame({
        'subject': [1, 1, 1, 1, 1, 1,
                    2, 2, 2, 2, 2, 2],
        'list': [1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1],
        'trial_type': ['study', 'study', 'study',
                       'recall', 'recall', 'recall',
                       'study', 'study', 'study',
                       'recall', 'recall', 'recall'],
        'position': [1, 2, 3, 1, 2, 3,
                     1, 2, 3, 1, 2, 3],
        'item': ['absence', 'hollow', 'pupil',
                 'hollow', 'pupil', 'empty',
                 'fountain', 'piano', 'pillow',
                 'pillow', 'fountain', 'pillow'],
        'item_index': [0, 1, 2, 1, 2, np.nan,
                       3, 4, 5, 5, 3, 5],
        'task': [1, 2, 1, 2, 1, np.nan,
                 1, 2, 1, 1, 1, 1],
        'distract': [1, 2, 3, np.nan, np.nan, np.nan,
                     3, 2, 1, np.nan, np.nan, np.nan],
    })
    return data


def param_def():
    param_def = parameters.Parameters()
    param_def.add_fixed(
        B_rec=0.8,
        B_start=0,
        Afc=0,
        Dfc=1,
        Acf=0,
        Dcf=1,
        Lfc=1,
        Lcf=1,
        P1=0,
        P2=1,
        T=10,
        X1=0.05,
        X2=1
    )
    return param_def


def test_cmr(data):
    model = cmr.CMR()
    param_def = parameters.Parameters()
    param_def.fixed = {'B_enc': .5, 'B_rec': .8,
             'Afc': 0, 'Dfc': 1, 'Acf': 0, 'Dcf': 1,
             'Lfc': 1, 'Lcf': 1, 'P1': 0, 'P2': 1,
             'T': 10, 'X1': .05, 'X2': 1}
    logl, n = model.likelihood(data, param_def)
    return logl, n

def test_cmr_fit(data, param_def):
    model = cmr.CMR()
    param_def.add_free(B_enc=(0, 1))
    results = model.fit_indiv(data, param_def, n_jobs=2)
    return results

def test_generate(data):
    data = data.copy()
    rec = TestRecall()
    study = data.loc[data['trial_type'] == 'study']
    subj_param = {1: {'recalls': [1, 2]},
                  2: {'recalls': [2, 0, 1]}}
    sim = rec.generate(study, {}, subj_param_fixed=subj_param)
    expected = ['absence', 'hollow', 'pupil',
                'hollow', 'pupil',
                'fountain', 'piano', 'pillow',
                'pillow', 'fountain', 'piano']
    return sim

def test_generate_subject(data):
    data = data.copy()
    rec = TestRecall()
    study = data.loc[(data['trial_type'] == 'study') &
                     (data['subject'] == 1)]
    param_def = parameters.Parameters()
    param_def.fixed = {'recalls': [1, 2]}
    # our "model" recalls the positions indicated in the recalls parameter
    rec_list = rec.generate_subject(study, {}, param_def)
    data_sim = fit.add_recalls(study, rec_list)
    expected = ['absence', 'hollow', 'pupil', 'hollow', 'pupil']
    return data_sim

this_data = data()
this_pdef = param_def()
logl, n = test_cmr(this_data)

res = test_cmr_fit(this_data, this_pdef)

this_data = data2()

sim = test_generate(this_data)

sim2 = test_generate_subject(this_data)

print('hi')