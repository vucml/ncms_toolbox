cd ..
q
e 3 versions of Minerva 2 I've made:

Minerva-ish uses the trace concept from Minerva 2 to perform free recall,
but it doesn't implement any of the echo or activation stuff. Instead, it
just calculates the Hamming distance between the probe and each trace in
the stack, then probabilistically chooses an item to recall based on the
item's Hamming distances.
    There are three different drift functions I've created. Drift randomly
changes elements in context. The probability of an element being changed is
the probability of being chosen (PC) times the activation level (AL).
BiasedDrift randomly chooses elements in the context array to be changed to
the element in that index in the previous word array. ForgettingDrift
randomly changes elements in context to 0. All of these drift mechanisms
have very similar effects on the lagcrp graph, so they're more or less
interchangeable.
    There are a couple functions that attempt to create category items,
but I didn't get very far on that in this version of Minerva.

                            . . . . . . . .

The second version of Minerva more closely resembles Hintzman's Minerva 2.
It uses a trace stack like before, but instead of using Hamming distances
to perform free recall, it converts the activation level of each trace
(given a probe) to weights, then uses those weights to probabilistically
choose an item to recall. This version also implements a stop probability
using an intensity threshold. If the intensity of the memory stack's echo
falls below a certain level, recall will be terminated. I managed to
calculate log likelihoods of the recall sequences as well, but I don't know
if those likelihoods are nonsensical. 

                            . . . . . . . . 

The third version of Minerva attempts to simulate the recognition
experiment from Hintzman's 1988 paper. It also conforms to the more
flexible structure of CMR-L, using an array of event structures and a 
switch case. Currently it can create category items, scramble them in a
list with filler items without having adjacent items of the same category,
and load the items into a stack of traces one by one. I didn't get the
recognition part running before I had to go back to school.

I hope this was helpful,
Stacey Xiang