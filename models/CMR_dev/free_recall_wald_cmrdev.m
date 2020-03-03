function [out_obj, env, net] = free_recall_wald_cmrdev(this_event, env, net, param, var_param)
%
% [this_logl, net] = free_recall_wald_cmrdev(this_event, env, net, param, var_param)
%
% [synth_event, net] = free_recall_wald_cmrdev(this_event, env, net, param, var_param)
%
% Requires wald-specific parameters / arguments
% param.n_racers
% param.range_rt
% param.alpha
% param.sigma
% param.drift <- strength
% if predictive:
% thisresp
% thisrt
%

% Top part of the code should be very similar to free_recall_word
% if param.verbose
%   fprintf('.');
% end

% this version continually reactivates start-list context
% prior to each recall attempt
if isfield(param, 'B_s') && param.B_s > 0
  net = start_context_reactivation_cmrdev(this_event, env, net, ...
                                          param, var_param);  
end

% some tension here, the code allows items from prior lists
% to enter the competition, but repeats are sometimes disallowed,
% have to grapple with adding CMR2 innovations in to the code

%AMIN = 0.000001;
%PMIN = 0.000001;

% three matrices can potentially influence f_in
net.f_in = zeros(size(net.f));

% experimental context
net.f_in = net.f_in + (net.w_cf * net.c);

% pre-experimental context
net.f_in = net.f_in + (net.w_cf_pre * net.c);

% item-to-item associations
% usually only effective if there's residual item activity on f
net.f_in = net.f_in + (net.w_ff_pre * net.f);

% if distributed representations are allowed, need to explicitly
% determine how much input there is for each accumulator

recallable_patinds = env.itemno_to_patind(:,2);
net.strength = zeros(1,length(recallable_patinds));


if strcmp(param.pattern_type,'unit_vec')
  net.strength = net.f_in(recallable_patinds)';
else
  for i=1:length(recallable_patinds)
    %net.strength(i) = dot(net.f_in, env.pattern(recallable_patinds(i),:)');
    % this way is much faster
    net.strength(i) = net.f_in' * env.pattern(recallable_patinds(i),:)';
  end
end

% does every decision unit need to have some small amount of
% strength?
% net.strength(net.strength<AMIN) = AMIN;

% sampling rule allows strength to be perturbed
switch param.sampling_rule
  case 'power'

    net.strength = net.strength .^ param.T;
 
  case 'classic'
    % note: if strength is zero min transformed strength is 1, not sure I
    % realized that; this could be awkward when dealing with multiple
    % lists, check CMR2 paper.  Also HK02 sez this version only
    % holds when f are orthonormal
    net.strength = exp((2*net.strength) ./ param.T);
    
  case 'logistic'
    net.strength = 1 ./ (1 + exp(-1*param.k .* (net.strength-param.xz)));

  otherwise
    error('unspecified sampling rule');
end

% under consideration
if param.norm_f_in ~= 0
  norm_factor = 1 + ((norm(net.strength) - 1) * param.norm_f_in);
  net.strength = net.strength ./ norm_factor;
end

if param.lat_inh ~= 0
  net.strength_preinh = net.strength;
  max_str = max(net.strength);
  trans = (net.strength ./ max_str) * param.lat_inh;
  max_trans = max(trans);
  net.strength = max_str .* (exp(trans) ./ exp(max_trans));
end


% don't need a stop prob, since recalls take time; do need to keep
% track of time.  don't create a prob model, since probs are continuous

% now we have all the strengths, can set up the call to the wald racers function



% Note: if you give every strength value a racer, then things with
% strength 0 can still technically win since it is a diffusion process
wald_param.n_racers = length(net.strength);
wald_param.range_rt = [0 env.rec_duration-env.rec_clock];
% wald_param.alpha = net.alpha;
wald_param.sigma = param.wald_sigma;
wald_param.drift = net.strength';
wald_param.drift(wald_param.drift<param.wald_min_drift) = param.wald_min_drift;

% exclude here really means 'inhibit'
% this is temporary inhibition in that it doesn't get stored on net
if param.exclude_repeats
  for i=1:length(env.previous_recalls)
    % convert to a pat ind
    temp_patind = env.itemno_to_patind(env.previous_recalls(i)==env.itemno_to_patind(:,1),2);
    % put it in an inhibited state
    net.alpha(temp_patind==recallable_patinds) = ...
        param.wald_alpha + param.wald_alpha_inhib;
  end
end
wald_param.alpha = net.alpha;

%if param.exclude_unseen
% like a priming mechanism, the default threshold for a word is
% high, and is lower for things that have been seen recently.
% Could implement this by making alpha a property of the net
%end

% if predictive, determine the probability; if generative,
% determine who wins the competition
this_pat_ind = [];

switch param.simulation_mode

  case 'predictive'

    % call the wald function to get a probability
    wald_param.simulation_mode = 'predictive';
    
    if strcmp(this_event.type,'FREE_REC_WORD')
      % which response was made, given as an index relative to the
      % set of strength values
      wald_in.thisresp = find(this_event.pat_ind==recallable_patinds);
      % have to make sure the recall events have time on them
      % CHECK THIS
      wald_in.thisrt = this_event.time - env.decision_start_time;
    elseif strcmp(this_event.type,'FREE_REC_END')
      wald_in.thisresp = -1;
      % wald code will set thisrt to max from range_rt
      wald_in.thisrt = 0;
    elseif strcmp(this_event.type,'CUED_REC_RESP')
      % set up time here!
      if this_event.itemno == -1
        wald_in.thisresp = -1;
        wald_in.thisrt = 0;
      else
        wald_in.thisresp = find(this_event.pat_ind==recallable_patinds);
        wald_in.thisrt = this_event.rt;
      end
      
    end
    
    wald_out = wald_decision(wald_in, wald_param);
    this_logl = wald_out.logl;
    out_obj = this_logl;
    
    this_pat_ind = this_event.pat_ind;
    this_itemno = this_event.itemno;
    
    % update the clock for the next event
    env.decision_start_time = this_event.time;
    env.rec_clock = this_event.time;
    
    if param.catch_impossible && ...
          (isinf(this_logl) || isnan(this_logl) || imag(this_logl)~=0)
      keyboard;
    end
    
  case 'generative'
   
    % sample an event using wald decision process
    wald_param.simulation_mode = 'generative';
    wald_in = struct();
    wald_out = wald_decision(wald_in, wald_param);
    % keyboard
    % figure out what item the response corresponds to
    if wald_out.resp == -1
      
      % create a stop event
      ev = struct();
      ev.subject = this_event.subject;
      ev.session = this_event.session;
      ev.trial = this_event.trial;
      ev.phase = this_event.phase;
      ev.type = 'FREE_REC_END';
      % latency models can use this, env should track time
      %ev.time = [];
      ev = propval(ev, env.default_event_fields);

      out_obj = ev;
      
    else
      
      % convert event ind to a pat ind
      this_pat_ind = recallable_patinds(wald_out.resp);
      this_itemno = env.itemno_to_patind(env.itemno_to_patind(:,2)==this_pat_ind,1);
      
      % update the environment recall timer
      env.rec_clock = env.rec_clock + wald_out.rt;
      % set the prev_response time 
      env.decision_start_time = env.decision_start_time + wald_out.rt;
      
      % create a recall event
      ev = struct();
      ev.subject = this_event.subject;
      ev.session = this_event.session;
      ev.trial = this_event.trial;
      ev.phase = this_event.phase;
      ev.type = 'FREE_REC_WORD'; 
      % need itemno
      ev.itemno = this_itemno;
      if isfield(env,'catno_to_patind')
        ev.category = env.catno_to_patind(env.catno_to_patind(:,2)==this_pat_ind,1);
      end
      ev.time = env.rec_clock;
      ev.rt = wald_out.rt;
      % let the events2data script deal with spos 
      %ev.serial_position = [];
      %ev.time = [];
      % will the recall event need to store pat_ind? once the
      % simulation is done I think this field is not
      % informative. can add back in if needed.
      % ev.pat_ind = recallable_patinds(event_ind);
      ev = propval(ev, env.default_event_fields);
      
      out_obj = ev;
      %keyboard
    end
    
end
    
% the consequences of the recall event
% reactivation code

% logging and simulating the recall event 
% if strcmp(ev.type,'FREE_REC_WORD')
% this_pat_ind can be empty 
if this_pat_ind > 0

  env.previous_recalls(end+1) = this_itemno;
  env.output_position = env.output_position + 1;

  % reactivate the recalled item
  net.f(:) = env.pattern(this_pat_ind,:);
  
  % update context based on what was recalled
  net.c_in = net.w_fc * net.f;
  net.c_in = net.c_in + net.w_fc_pre * net.f;
  net.c_in = normalize_vector(net.c_in);
  
  % update context
  rho = sqrt(1 + param.B^2 * ((dot(net.c,net.c_in)^2) - 1)) - ...
        param.B * dot(net.c,net.c_in);
  net.c = rho * net.c + param.B * net.c_in;
  
  % learning
  % update weights
  net.w_fc = net.w_fc + ((net.c * net.f') .* param.rec_lrate_fc);
  net.w_cf = net.w_cf + ((net.f * net.c') .* param.rec_lrate_cf);

end

% if this_event.trial==2
% keyboard
% end
