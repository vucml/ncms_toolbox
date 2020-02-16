import numpy as np
import matplotlib.pyplot as plt

# some helper functions

# make a serial position curve
def quick_spc(recalls, list_length):
    # assumes clean recalls: no repeats or intrusions
    n_trials = recalls.shape[0]
    numer = np.zeros(list_length)
    for i in range(list_length):
        # serial positions start at 1
        this_spos = i + 1
        # how many times does this spos appear
        numer[i] = np.sum(recalls==this_spos)
    spos_prec = numer / n_trials
    return spos_prec

def quick_plot_spc(spos_prec):
    fig = plt.figure()
    plt.plot(range(len(spos_prec)),spos_prec,'ko-')
    plt.show()


