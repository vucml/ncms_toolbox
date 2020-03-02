function param = update_param(param,var_param,env)
% UPDATE_PARAM
%
% Inputs:
% param.  The standard param structure
%
% var_param.  p_idx - structure of which parameters will be
%                     modified and each parameter's index in param_mat
%             param_mat - matrix of trials x events x parameters
%
% env.  Two fields, env.trial, and env.event
%

mod_params = fieldnames(var_param.p_idx);

for cur_param = 1:length(mod_params)
  
  param.(mod_params{cur_param}) = var_param.param_mat(env.trial, ...
                                                    env.event, ...
                                                    cur_param);

end
