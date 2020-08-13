
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from psifr import fr

from cfr import task
from cfr import figures

# gotta load the orig data and check that the effect is there

sim_file = '/Users/polyn/Science/Analysis/catFR/dropbox/v3/cmr_fcf-loc-cat-use/sim.csv'
fig_dir = '/Users/polyn/Science/Analysis/catFR/dropbox/v3/cmr_fcf-loc-cat-use/'
data_dir = '/Users/polyn/Science/Analysis/catFR/dropbox/'
data_fig_dir = '/Users/polyn/Science/Analysis/catFR/figs/'
data_file = 'cfr_eeg_mixed.csv'

print('read free recall data')
# sim = task.read_free_recall(sim_file)
data = task.read_free_recall(data_dir+data_file)

# sim['category'] = sim['category'].astype('category')
# sim.category.cat.as_ordered(inplace=True)

print('plot crps empirical')

# all transitions

categories = data['category'].cat.categories
cat_crp = [fr.lag_crp(data, item_query=f'category == "{category}"')
           for category in categories]
crp = pd.concat(cat_crp, keys=categories, axis=0)
crp.index = crp.index.set_names('category', level=0)

g = fr.plot_lag_crp(crp, hue='category')
g.add_legend()
g.set(ylim=(0, .5))
g.savefig(data_fig_dir+'m13_within_cat_by_cat.pdf')

# remove last few serial positions
cat_crp2 = [fr.lag_crp(data, item_query=f'(input < 22) & (category == "{category}")')
           for category in categories]
crp2 = pd.concat(cat_crp2, keys=categories, axis=0)
crp2.index = crp2.index.set_names('category', level=0)

h = fr.plot_lag_crp(crp2, hue='category')
h.add_legend()
h.set(ylim=(0, .7))
# input position less than 22
h.savefig(data_fig_dir+'m13_within_cat_by_cat_ip_lt22.pdf')

# remove first few output positions
cat_crp3 = [fr.lag_crp(data, item_query=f'(output > 1) & (category == "{category}")')
           for category in categories]
crp3 = pd.concat(cat_crp3, keys=categories, axis=0)
crp3.index = crp3.index.set_names('category', level=0)

h = fr.plot_lag_crp(crp3, hue='category')
h.add_legend()
h.set(ylim=(0, .7))
# output position greater than 1
h.savefig(data_fig_dir+'m13_within_cat_by_cat_op_gt1.pdf')


print('plot crps for sim')

# categories = sim['category'].cat.categories
# cat_crp = [fr.lag_crp(sim, item_query=f'category == "{category}"')
#            for category in categories]
# crp = pd.concat(cat_crp, keys=categories, axis=0)
# crp.index = crp.index.set_names('category', level=0)
#
# g = fr.plot_lag_crp(crp, hue='category')
# g.add_legend()
# g.set(ylim=(0, .5))

# plt.savefig(sim_fig_dir+'sim_within_cat_by_cat.pdf')

# figures.plot_fit(
#         sim, 'category', 'lag_crp_within', fr.lag_crp,
#         {'test_key': 'category', 'test': lambda x, y: x == y},
#         'prob', fr.plot_lag_crp, {}, fig_dir
#     )
# load the data
# df = pd.read_csv(data_file, dtype={'category': 'category'})
# df.category.cat.as_ordered(inplace=True)

# study = df.query('trial_type == "study"').copy()
# recall = df.query('trial_type == "recall"').copy()
# data = fr.merge_lists(study, recall,
#                      list_keys=['list_type', 'list_category'],
#                      study_keys=['category'])

print('hi')

