




# This works!
# tester_fixed = {'B_rec': 0.5, 'PX': 8, 'neural_scaling': 0.1}
# tester_rec = {'input': np.array([0, 1, 2], dtype=int)}
# gen_dynamic = {'recall': {'B_rec': 'clip(B_rec + random.randn() * neural_scaling, 0, 1)'}}
# tester_dynam = parameters.set_dynamic(tester_fixed, tester_rec, gen_dynamic['recall'])

# param_def.add_fixed(B_enc=0.5)
# param_def.fixed['B_enc']

# if you want a dynamic recall param, but do not need to recover the sequence of random numbers:
# gen_dynamic = {'recall': {'B_rec': 'clip(B_rec + random.randn() * neural_scaling, 0, 1)'}}
# this tells the code that the synthetic neural signal (the 'hcmp' column on the data structure)
# is attached to the study events. Generative simulations create recall events, so they don't
# take recall events as inputs. For this version of the model we aren't allowing errors,
# so the max number of recall events is the same as the number of study events.
# As such, if you have a generative simulation with a dynamic parameter that changes
# from recall event to recall event, and requires externally provided values,
# the dynamic parameter evaluation code will check the study structure for those
# values. It will use 'position' on the study events to reference output position of
# the recall events.

# Neal raised the point that this is an awkward way to control dynamic recall
# parameters. Alternative could be to create dummy recall events that are part
# of the data structure first argument to model.generate.  Then generate_subject
# could have an optional input keyword argument recall_data=None.  If it is
# None, the code operates as it does presently.  If recall_data exists, it would be
# converted to list format like study events, and consulted when a dynamic recall
# parameter exists

# When you run the generative simulation with a dynamic recall parameter
# you lose the hcmp field.  This code copies over the hcmp value from the original
# set of dummy recalls (on synth_study2)

for index, row in dyn_sim.iterrows():

    if row['trial_type']=='recall':
        # filter to get the dummy recall event with this
        # subject, list, position

        m1 = synth_study2['subject']==row['subject']
        m2 = synth_study2['list']==row['list']
        m3 = synth_study2['trial_type']=='recall'
        m4 = synth_study2['position']==row['position']
        mask = m1 & m2 & m3 & m4
        # mask should return a single value
        val = synth_study2['hcmp'][mask]
        dsh.loc[index,('hcmp')] = val.to_numpy()[0]


# We simulate the neural signal as a stochastic process producing values
# drawn from a normal distribution with mean = 0 and stdev = 1.
var_signal = np.random.randn(ndf.shape[0])
# then we create a column on the dataframe called 'hcmp' (short
# for hippocampus)
ndf = ndf.assign(hcmp=pd.Series(var_signal).values)
# for these simulations we only want to keep the values associated with
# the recall events, can set the hcmp values for study events to be 'missing'
ndf.loc[ndf['trial_type']=='study', 'hcmp'] = np.nan

