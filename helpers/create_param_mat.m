function vp = create_param_mat(var_param,numTrials,listLength,recallLength)
% CREATE_PARAM_MAT
%
% Alters var_param structure of one form, to a more efficient form
%
% Inputs:
% param.  The standard param structure
% var_param.  Fields: vp.name, vp.update_level, vp.val
% env.  Two fields, env.trial, and env.event
% update_level.  Options: 'study_event', 'recall_event', 'trial'
%
%

% initialize param mat, assuming it is of size trials x listLength + recalls
% + 1 for stop position
% x number of modified parameters

param_mat = nan(numTrials, listLength + recallLength + 1, length(var_param));

p_idx = struct;

for v = 1:length(var_param)
  
  p_idx = setfield(p_idx, var_param(v).name, v);

  if strcmp(var_param(v).update_level, 'recall_event')
    param_mat(:, listLength+1:listLength+recallLength, v) = var_param(v).val;
  elseif strcmp(var_param(v).update_level, 'study_event')
    param_mat(:, 1:listLength, v) = var_param(v).val;
  elseif strcmp(var_param(v).update_level, 'trial')
    param_mat(:,:,v) = repmat(var_param(v).val, 1, size(param_mat,2));
  end  
  
end

vp.param_mat = param_mat;
vp.p_idx = p_idx;