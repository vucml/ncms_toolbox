function p = p_recall_cmr(w_cf, c, LL, prev_rec, output_pos, param, w_cf_pre)
%P_RECALL_CMR   Probability of recall according to CMR.
%
%  p = p_recall_cmr(w_cf, c, LL, prev_rec, output_pos, param)
%
%  INPUTS:
%        w_cf:  [list length+1 X list length+1] matrix of
%               context-to-item associative weights.
%
%           c:  [list length+1 X 1] vector indicating the state of
%               context to use as a cue.
%
%          LL:  list length.
%
%    prev_rec:  vector of the serial positions of previous recalls.
%
%  output_pos:  output position (the number of items previously
%               recalled; the first recall attempt is 0).
%
%       param:  structure with model parameter values.
%
%  OUTPUTS:
%        p:  [1 X list length+1] vector of recall event probabilities;
%            p(LL+1) is the probability of stopping.

AMIN = 0.000001;
PMIN = 0.000001;

% control process for recall initiation
% only check for first recall (no prev recalls)
% if it exists and is set to true
if isempty(prev_rec)
  if isfield(param,'initiation_control') && param.initiation_control
    % this is a temporary change as param won't get passed out
    param.sampling_rule = 'logistic';
  end
end

% determine cue strength
if isfield(param, 'I') && param.I ~= 0 && ...
      (~isempty(prev_rec) || param.init_item)
  % experimental cuing strength
  strength_exp = (w_cf * c)';
  f = zeros(size(c));
  if isempty(prev_rec)
    unit = LL;
  else
    unit = prev_rec(end);
  end
  f(unit) = 1;
  
  % pre-exp cuing strength
  pre_exp_cue = param.I * f + (1 - param.I) * c;
  strength_pre = (w_cf_pre * pre_exp_cue)';
  f_in = strength_exp + strength_pre;

elseif isfield(param, 'I') && param.I == 1 && ...
      (isempty(prev_rec) && ~param.init_item)
  f_in = (w_cf * c)';
else
  f_in = ((w_cf + w_cf_pre) * c)';
end

f_in = f_in(1:LL);
f_in(f_in < AMIN) = AMIN;

switch param.sampling_rule
  case 'power'
    if isfield(param, 'ST') && param.ST ~= 0
      remaining = 1:LL;
      remaining = remaining(~ismember(remaining, prev_rec));
      s = sum(f_in(remaining));
      % if isempty(prev_rec)
      %   s = sum(strength) / min(strength);
      % else
      %   s = sum(strength(remaining)) / sum(strength(prev_rec));
      % end
      param.T = param.T * (s^param.ST);
    end
    strength = f_in .^ param.T;
 
  case 'classic'
    strength = exp((2*f_in) ./ param.T);
    
  case 'logistic'
    strength = 1 ./ (1 + exp(-1*param.k .* (f_in-param.xz)));

    % buffer read-out control process, only with logistic sampling
    if isfield(param,'control_proc') && param.control_proc
      % did you already recall something?
      if ~isempty(prev_rec)
        % aside from what's already been recalled, are there logistically supercharged items?
        temp_strength = strength;
        temp_strength(prev_rec) = [];
        if any(temp_strength > 0.96)
          % this triggers the control process
          % amplify the most recent recall's context element
          amp_vec = zeros(size(c));
          amp_vec(prev_rec(end)) = param.c_amp;
          temp_c = (c + amp_vec) ./ norm(c + amp_vec);
          % update f_in, don't need sem wts here
          f_in = (w_cf * temp_c)';
          f_in = f_in(1:LL);
          f_in(f_in < AMIN) = AMIN;
          % update strength
          strength = f_in .^ param.c_amp_exp;
        end
      end
    end % control process

  otherwise
    error('unspecified sampling rule');
end

% if ~isempty(prev_rec) && prev_rec(1) == 17
%   keyboard
% end
if param.lat_inh ~= 0
  strength_preinh = strength;
  max_str = max(strength);
  trans = (strength ./ max_str) * param.lat_inh;
  max_trans = max(trans);
  strength = max_str .* (exp(trans) ./ exp(max_trans));
end
  
if sum(strength(1:LL)) == 0
  % if strength is zero for everything, set equal support for everything
  strength(1:LL) = 1;
end

% set activation of previously recalled items to 0
strength_all = strength;
strength(prev_rec) = 0;

% stop probability
p = NaN(1, LL+1);
p(end) = p_stop_cmr(output_pos, prev_rec, strength_all, param, PMIN);

if p(end) == 1
  % if stop probability is 1, recalling any item is impossible
  p(1:LL) = 0;
else
  % recall probability conditional on not stopping
  p(1:LL) = (1 - p(LL+1)) .* (strength ./ sum(strength));
end

if any(isnan(p))
  % sanity check in case some weird case comes through in the data
  % that the code wasn't expecting
  error('Undefined probability.')
end

% if ~isempty(prev_rec)
%   if ismember(prev_rec(end),[5:11])
%     keyboard
%   end
% end

function rho = scale_context(cdot, B)

rho = sqrt(1 + B^2 * (cdot^2 - 1)) - (B * cdot);

