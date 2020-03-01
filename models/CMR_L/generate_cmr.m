function seq = generate_cmr(param, data, var_param, n_rep)
%GENERATE_CMR   Generate simulated data from CMR.
%
%  Same as cmr_general, but generates recall sequences rather than
%  calculating the probability of data given the model.
%
%  param and data are assumed to be pre-processed, including setting
%  defaults for missing parameters, etc.
%
%  seq = generate_cmr(param, data, var_param, n_rep)
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
  [f, c, w_fc, w_cf, w_cf_pre, env] = init_network_cmr(param, env, pres_itemnos);
  
  % study
  [f, c, w_fc, w_cf, env] = present_items_cmr(f, c, w_fc, w_cf, param, ...
                                         var_param, env, LL);
  
  % recall
  stopped = false;
  pos = 0;
  seq = [];
  while ~stopped
      
    if ~isempty(var_param)
      param = update_param(param, var_param, env);
    end
    
    if isfield(param, 'B_s') && param.B_s > 0
      % at end of list, assume some of start list context is pushed into
      % context
      s_index = env.s_unit;
      
      % present item
      f(:) = 0;
      f(s_index) = 1;

      if pos==0 && isfield(param,'B_s_init')
        
        if param.B_s_init_transient
          % save the context state
          c_state = c;
        end

        % update context
        rho = scale_context(dot(c, f), param.B_s_init);
        c = rho * c + param.B_s_init * f;
      else
        % update context
        rho = scale_context(dot(c, f), param.B_s);
        c = rho * c + param.B_s * f;
      end
    end
    
    % recall probability given associations, the current cue, and
    % given that previously recalled items will not be repeated
    prob_model = p_recall_cmr(w_cf, c, LL, seq, pos, param, w_cf_pre);

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

    % the idea is if you didn't actually recall the first item then
    % context proceeds as if you hadn't folded that big burst of
    % start-context into it.
    if pos==1 && ~stopped && param.B_s_init_transient
      if seq(1) ~= 1
        c = c_state;
      end
    end
    
    if ~stopped
      % reactivate the item and reinstate context
      [f, c] = reactivate_item_cmr(f, c, w_fc, event, param);
    end

    %update event
    env.event = env.event + 1;
    
  end

function rho = scale_context(cdot, B)

rho = sqrt(1 + B^2 * (cdot^2 - 1)) - (B * cdot);
