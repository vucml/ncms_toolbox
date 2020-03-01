function [logl, logl_all] = cmr_general(param, data, var_param)
%CMR_GENERAL   Calculate log likelihood for free recall using CMR.
%
%  Calculates log likelihood for multiple lists. param and data are
%  assumed to be pre-processed, including setting defaults for
%  missing parameters, etc.
%
%  [logl, logl_all] = cmr_general(param, data, var_param)
%
%  INPUTS:
%   param:  structure with model parameters. Each field must contain a
%           scalar or a string. See README.txt for list of parameters.
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
%  OUTPUTS:
%      logl:  [lists X recalls] matrix with log likelihood values for
%             all recall events in data.recalls (plus stopping events).
%
%  logl_all:  [lists X recalls X events] matrix of log likelihood values
%             for all possible events, after each recall event in
%             data.recalls.

% if var_param isn't specified, set it as an empty array, so the
% code can quickly determine that it is irrelevant
if nargin < 3
  var_param = [];
end

% this function takes the existing param structure and runs some
% basic checks, mostly regarding whether all necessary parameters
% are specified / exist
param = check_param_cmr(param);

% we use the data structure to determine how many trials we will be
% simulating, how many items are presented on each list, and the
% maximum number of recall events we will be simulating
[n_trials, n_items] = size(data.pres_itemnos);
n_recalls = size(data.recalls,2);

% we initialize the log-likelihood fitness scores as NaNs, these
% will be overwritten as trials are simulated.  an extra element is
% added to n_recalls to simulate the stop event.  logl will contain the
% likelihood of each observed event, while logl_all will also
% contain the likelihoods of the other possible events.  when a
% given trial has fewer recalls than n_recalls, the remaining
% elements in logl (the padding) will be NaN
logl = NaN(n_trials, n_recalls + 1);
logl_all = NaN(n_trials, n_recalls + 1, n_items + 1);

% we iterate through each simulated trial, calling the embedded
% run_trial function. cmr_general assumes that each trial is
% independent of the others; in other words, the network is
% re-initialized prior to each trial 
for i = 1:n_trials
  
  % the env structure is used to keep track of which trial this is,
  % and which event we are simulating within the trial
  env.trial = i;
  env.event = 1;
  
  % var_param contains a list of which parameters may be updated,
  % and a schedule describing what values they will take, using
  % env.trial and env.event to index that schedule
  if ~isempty(var_param)
    param = update_param(param,var_param,env);
  end
  
  % the actual work is done by this sub-function
  % var_param: passed in case there are parameters that vary
  % within the trial.
  [logl_trial, logl_all_trial] = run_trial(param, ...
                                           var_param, ...
                                           env, ...
                                           data.pres_itemnos(i,:), ...
                                           data.recalls(i,:));
  
  % bookkeeping to capture the likelihood scores produced by the simulation
  ind = 1:length(logl_trial);
  logl(i,ind) = logl_trial;
  logl_all(i,ind,:) = logl_all_trial;
  
end

  
function [logl, logl_all] = run_trial(param, var_param, env, pres_itemnos, recalls)

  % list length, number of serial positions in the study list.  the
  % assumption is that all trials have the same length 
  LL = size(pres_itemnos, 2);
  
  % recalls contains the serial position identities of the recalled
  % items.  the final element represents recall termination, and is
  % given a code of LL+1
  seq = [nonzeros(recalls)' LL + 1];
  
  % initialize the model
  [f, c, w_fc, w_cf, w_cf_pre, env] = init_network_cmr(param, env, pres_itemnos);
  
  % SMP: NOTE: these don't seem to be used anywhere
  w_fc_pre_s = w_fc;
  w_cf_pre_s = w_cf;

  % this function simulates the study period: the presentation of
  % the items, as well as the potential for inter-item distraction
  [f, c, w_fc, w_cf, env] = present_items_cmr(f, c, w_fc, w_cf, param, ...
                                              var_param, env, LL);

  logl = zeros(size(seq));
  logl_all = NaN(length(seq), LL+1);
  for i = 1:length(seq)
           
    if ~isempty(var_param)
      param = update_param(param,var_param,env);
    end
    
    if isfield(param, 'B_s') && param.B_s > 0
      % at end of list, assume some of start list context is pushed into
      % context
      s_index = env.s_unit;
      
      % present item
      f(:) = 0;
      f(s_index) = 1;

      if i==1 && isfield(param,'B_s_init')

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
    
    % probability of all possible events
    output_pos = i - 1;
    prev_rec = seq(1:output_pos);
    prob_model = p_recall_cmr(w_cf, c, LL, prev_rec, output_pos, ...
                              param, w_cf_pre);
    
    % calculate log likelihood for actual and possible events
    logl(i) = log(prob_model(seq(i)));
    logl_all(i,:) = log(prob_model);
    
    % the idea is if you didn't actually recall the first item then
    % context proceeds as if you hadn't folded that big burst of
    % start-context into it.
    if i==1 && param.B_s_init_transient
      if seq(1) ~= 1
        c = c_state;
      end
    end

    % if all(isnan(logl_all(i,:)))
    %   keyboard
    % end

    if i < length(seq)
      % reactivate the item and reinstate context
      [f, c] = reactivate_item_cmr(f, c, w_fc, seq(i), param);
    end
    
    env.event = env.event+1;
  end

function rho = scale_context(cdot, B)

rho = sqrt(1 + B^2 * (cdot^2 - 1)) - (B * cdot);
