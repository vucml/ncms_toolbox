function seq = generate_minerva(param, data, var_param, n_rep)
%GENERATE_MINERVA   Generate simulated data from MINERVA.
%
%
%  param and data are assumed to be pre-processed, including setting
%  defaults for missing parameters, etc.
%
%  seq = generate_minerva(param, data, var_param, n_rep)
%
%
%  INPUTS:
%   param:  structure with model parameters. Each field may contain a
%           scalar or a vector. Vector fields indicate that different
%           parameters are used for different lists. For field f, trial
%           i, the value used is: param.(f)(index(i))
%           or just param.(f) if f is a scalar field or if index is
%           omitted.
%
%    data:  free recall data structure, with repeats and intrusions
%           removed. Required fields:
%            recalls
%            pres_itemnos
%
% var_param: structure with information about parameters that vary
%            by trial, by study event, or by recall event.
%            Required fields:
%             name
%             update_level
%             val
%
%   n_rep:  number of times to replicate the experiment.
%
%  OUTPUTS:
%      seq:  [lists X recalls] matrix giving the serial positions of
%            simulated recalls. Stop events are not included, so that
%            the matrix is comparable to data.recalls.

if nargin < 4
  n_rep = 1; 
end

if ~exist('var_param')
  var_param = [];
end

param = check_param_cmr(param);

[n_trials, n_items] = size(data.pres_itemnos);
n_recalls = size(data.recalls,2);
seq = zeros(n_trials * n_rep, n_items);
n = 0;
for i = 1:n_rep
  for j = 1:n_trials
    % run a trial. Assuming for now that each trial is independent of
    % the others
    n = n + 1;

    env.trial = j;
    env.event = 1;

    if ~isempty(var_param)
      param = update_param(param, var_param, env);
    end

    seq_trial = run_trial(param, var_param, env, data.pres_itemnos(j,:));
    seq(n,1:length(seq_trial)) = seq_trial;
  
  end
end

function seq = run_trial(param, var_param, env, pres_itemnos)
  
  LL = size(pres_itemnos, 2);

  % initialize the model
  [env] = init_network_minerva(param, env, pres_itemnos);
  
  % study
  [f, c, w_fc, w_cf, env] = present_items_minerva(param, ...
                                         var_param, env);
  
  % recall
  stopped = false;
  pos = 0;
  seq = [];
  while ~stopped
      
    if ~isempty(var_param)
      param = update_param(param, var_param, env);
    end
    
    % recall probability given associations, the current cue, and
    % given that previously recalled items will not be repeated
    prob_model = p_recall_minerva(w_cf, c, LL, seq, pos, param, w_cf_pre);

    % sample an event
    event = randsample(1:(LL+1), 1, true, prob_model);

    if event == (LL + 1)
      % if the termination event was chosen, stop recalling
      stopped = true;
    else
      % record the serial position of the recall
      seq = [seq event];
      pos = pos + 1;
    end

    if ~stopped
      % reactivate the item and reinstate context
      [f, c] = reactivate_item_minerva(f, c, w_fc, event, param);
    end

    %update event
    env.event = env.event + 1;
    
  end


