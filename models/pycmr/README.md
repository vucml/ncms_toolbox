# pyCMR

Python-based object-oriented implementation of the Context Maintenance and Retrieval (CMR) model.  Currently in a very early stage of development.

## Notes
* See Projects and Issues for next-up to-do items
* scratch_ifr_pycmr.py: _Start here!_ Contains sample code for setting some basic parameters and running a bare-bones simulation of immediate free recall.  At a command line, you can try: python scratch_ifr_pycmr.py.  Has code to run the model in generative mode and then runs it in predictive mode, giving you the log-likelihood of the generated data. 
* ncms_model.py: Contains the Network, Layer, Projection class core functions.
* ncms_task.py: Contains the Task class and Item class core functions.
* ncms_classes.py: Contains overflow classes.  Note that I'm not sure about the ncms naming convention for all these files, trying to figure out where to use ncms, vs where to use pycmr.

## Task class functions
* __init__
* ifr_task_generate
* ifr_task_predict
* ifr_trial_generate
* ifr_trial_predict
* ifr_study_list
* ifr_recall_period_generate
* ifr_recall_period_predict

## Item class functions
* __init__

## Parameter class functions
* __init__

## Network class functions
* __init__
* add_layer
* add_projection
* initialize_basic_tcm
* present_item_basic_tcm
* initialize_context
* prob_recall_basic_tcm
* stop_function
* reactivate_item_basic_tcm

## Layer class functions
* __init__
* initialize_net_input_zeros
* normalize_net_input
* initialize_act_state_zeros
* activate_unit_vector
* integrate_net_input
* sampling_fn_classic
* update_activation (pass)

## Projection class functions
* __init__
* init_matrix_identity
* project_activity
* hebbian_learning
