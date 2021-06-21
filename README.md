# ncms_toolbox
Neuro-Cognitive Memory Search toolbox

This toolbox is maintained by Sean Polyn and the Vanderbilt Computational Memory Lab, and is designed to support the development and evaluation of models of neurocognitive processes in psychological memory search tasks. (Note: At this point the code focuses exclusively on the free-recall task.)  Many people have been involved in the development of this code, both on GitHub and on earlier internally maintained versions.  Much of the code in the toolbox is written in Matlab, but we now have links to companion code written in Python. Reach out if you need help making sense of the code, as the documentation is currently in a rather preliminary state.

Certain MATLAB analysis relies on the Episodic Memory Behavioral Analysis in Matlab toolbox, also hosted on GitHub: vucml/EMBAM.
Certain Python analysis relies on the psifr toolbox, also hosted on GitHub: mortonne/psifr & mortonne/psifr-notebooks.

* models/ directory contains separate directories for different modeling efforts
* helpers/ directory contains helper functions that aren't specific to a particular model
* search/ directory contains scripts and functions to do optimization that aren't specific to a particular model

## Models: CMR-L
* This directory contains a MATLAB version of the Context Maintenance and Retrieval (CMR) model of free recall used by Kragel et al. (2015) and based on Polyn et al. (2009).
* The L stands for "likelihood": CMR-L supports likelihood-based evaluation of free-recall data, as well as the generation of synthetic recall sequences.

## Models: cymr
* Neal Morton developed a Python version of CMR hosted on GitHub at mortonne/cymr. The cymr directory in this project contains code that is designed to work with cymr. (So you'll need to install cymr to get it to work!)
* Polyn (2021) Assessing neurocognitive hypotheses in a likelihood-based model of the free-recall task. In: Model-Based Cognitive Neuroscience, edited by Brandon Turner & Birte Forstmann. In ncms_toolbox > cymr you'll find tutorial code: KragEtal15_tutorial.py, which was written to accompany the chapter. Check the README in that directory for more details.

## References
* Kragel, J. E., Morton, N. W, and Polyn, S. M. (2015) Neural activity in the medial temporal lobe reveals the fidelity of mental time travel. Journal of Neuroscience, 35 (7), 2914-2926.
* Polyn, S. M., Norman, K. A., and Kahana, M. J. (2009a). A context maintenance and retrieval model of organizational processes in free recall. Psychological Review, 116(1), 129–156.
