# ncms_toolbox
Neuro-Cognitive Memory Search toolbox

This toolbox is maintained by Sean Polyn and the Vanderbilt Computational Memory Lab, and is designed to support the development and evaluation of models of neurocognitive processes in psychological memory search tasks.  Many people have been involved in the development of this code, both on GitHub and on earlier internally maintained versions.  Currently, the bulk of the toolbox is written in Matlab, but preliminary Python code (very preliminary) can be found within and is under development.  Reach out if you need help making sense of the code, as the documentation is currently in a rather preliminary state.

Certain analysis functions rely on the Episodic Memory Behavioral Analysis in Matlab toolbox, also hosted on GitHub: vucml/EMBAM.

* models/ directory contains separate directories for different modeling efforts
* helpers/ directory contains helper functions that aren't specific to a particular model
* search/ directory contains scripts and functions to do optimization that aren't specific to a particular model

## Models: CMR-L
* This directory contains both a predictive and generative version of the Context Maintenance and Retrieval (CMR) model of free recall used by Kragel et al. (2015) and based on Polyn et al. (2009).
* This version is written in Matlab.

## Models: pyCMR
* Under development, porting CMR-L to Python, and a more object-oriented framework.

## References
* Kragel, J. E., Morton, N. W, and Polyn, S. M. (2015) Neural activity in the medial temporal lobe reveals the fidelity of mental time travel. Journal of Neuroscience, 35 (7), 2914-2926.
* Polyn, S. M., Norman, K. A., and Kahana, M. J. (2009a). A context maintenance and retrieval model of organizational processes in free recall. Psychological Review, 116(1), 129–156.
