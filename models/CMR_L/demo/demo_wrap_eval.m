function out = demo_wrap_eval(param, neural_mat, fstruct)
% Helper function for demo code for Turner and Forstmann chapter
% 

% create a 'variable parameter' structure
% specifying that the B_rec param will change 
% from recall event to recall event, following the schedule 
% specified in the 'val' matrix
vp_neural = struct();
vp_neural(1).name = 'B_rec';
vp_neural(1).update_level = 'recall_event';
val = param.B_rec + neural_mat * param.n;
% allowable range of B_rec is 0 to 1 
val(val<0) = 0;
val(val>1) = 1;
vp_neural(1).val = val;

% task details
n_trials = size(fstruct.data.pres_itemnos, 1);
LL = size(fstruct.data.pres_itemnos, 2);
max_recalls = size(fstruct.data.recalls, 2);

% reconfigures the variable parameter structure to be more efficient
vp_neural_opt = create_param_mat(vp_neural, n_trials, LL, max_recalls);
fstruct.var_param = vp_neural_opt;

% run the predictive version of the model to get a log-likelihood
% technically the eval function returns -1 * log-likelihood
out.err = eval_param_cmr(param, fstruct);
out.fstruct = fstruct;

