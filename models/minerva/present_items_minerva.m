function [net, env] = present_items_minerva(net, env, param, var_param)
%PRESENT_ITEMS_MINERVA
%
%
% env.event
% env.LL

% stimulus generation
% orthonormal vectors
% randn vectors
% binary [0 1] / ternary [-1 0 1] 
% geometric, the set of positive integers, w or w/o 0
% 


% how carved?
% given a set of study events
% a set of well-formatted study events
% iterate through the set and do some bookkeeping
% create the storage matrix




% calls function for individual study event
% in object-oriented language, study_event class has associated

% function 'study_list'
%    % given set of study events, code will sort them by
%    % onset, call present_item for each, 
%    % model is componential, only passes needed components in?
%    % present_item, for each item, checks for function handle
%    specifying relevant present function
%    % so, distraction events have associated events as well


% 

for i = 1:LL
  % set event_specific parameters
  if ~isempty(var_param)
    param = update_param(param,var_param,env);
  end
  
  % interpresentation interval distraction
  if isfield(param, 'B_ipi')
    ipi_index = env.ipi_dist_unit(i);
    % present item
    % SMP: consult a list? 
    % SMP: this below is orthonormal codes constructed on the fly.
    % have to pass in a particular pattern
    f(:) = 0;
    f(ipi_index) = 1;
    
    % update context (assuming orthogonal item representations,
    % an orthogonal initial state of context, and no off-diagonal
    % pre-experimental associations on Mfc)
    rho = sqrt(1 - param.B_ipi^2);
    c = rho * c + param.B_ipi * f;
  end
  
  % activate item
  f(:) = 0;
  f(i) = 1;
  
  % update context
  if param.Afc == 0 && param.Sfc == 0
    % assuming orthogonal item representations, an orthogonal initial
    % state of context, and no off-diagonal pre-experimental
    % associations on Mfc
    rho = sqrt(1 - param.B_enc^2);
    c = rho * c + param.B_enc * f;
  else
    % must calculate the actual projection
    c_in = normalize_vector(w_fc(:,i));
    rho = scale_context(dot(c, c_in), param.B_enc);
    c = rho * c + param.B_enc * c_in;
  end
  
  % primacy
  P = (param.P1 * exp(-param.P2 * (i - 1))) + 1;

  % learning rate
  if isfield(param, 'L')
    P = P + param.L;
  end
  
  % update weights
  w_fc(:,i) = w_fc(:,i) + c;
  w_cf(i,:) = w_cf(i,:) + (P * c');

  % update event counter
  env.event = env.event + 1;
end

if isfield(param, 'B_ri');
  % end-of-list distraction
  ri_index = env.ri_dist_unit;  
  % present item
  f(:) = 0;
  f(ri_index) = 1;
  
  % update context (assuming orthogonal item representations,
  % an orthogonal initial state of context, and no off-diagonal
  % pre-experimental associations on Mfc)
  rho = sqrt(1 - param.B_ri^2);
  c = rho * c + param.B_ri * f;
end

gate = true;
if isfield(param, 'B_s_always') 
  if param.B_s_always == 1
    gate = false;
  end
end

if isfield(param, 'B_s') && gate
  % at end of list, assume some of start list context is pushed into
  % context
  s_index = env.s_unit;
  
  % present item
  f(:) = 0;
  f(s_index) = 1;
  
  % update context
  rho = scale_context(dot(c, f), param.B_s);
  c = rho * c + param.B_s * f;
end

function rho = scale_context(cdot, B)

rho = sqrt(1 + B^2 * (cdot^2 - 1)) - (B * cdot);







