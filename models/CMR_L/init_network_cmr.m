function [f, c, w_fc, w_cf, w_cf_pre, env] = init_network_cmr(param, env, pres_itemnos)
%INIT_NETWORK_CMR   Initialize network variables for CMR.
%
%  [f, c, w_fc, w_cf, w_cf_pre, env] = init_network_cmr(param, env, pres_itemnos)
%
%  INPUTS:
%         param:  structure of model parameters.
%
%     var_param:  which parameters are variable.  if B_ipi and B_ri
%                 exist, then extra units will be created. 
%           env:  structure to keep track of time and unit indices
%
%  pres_itemnos:  [lists X items] matrix of item numbers.
%
%  OUTPUTS:
%         f:  [list length+1 X 1] vector feature layer, with no units
%             activated.
%
%         c:  [list length+1 X 1] vector context layer, set to an
%             initial state of context orthogonal to all presented
%             items.
%
%      w_fc:  [list length+1 X list length+1] matrix of item-to-context
%             associative weights. Set to zeros, except for the
%             diagonal, which is 1 - param.G.
%
%      w_cf:  [list length+1 X list length+1] matrix of context-to-item
%             associative weights. Off-diagonal entries are set to
%             param.C, plus scaled semantic similarity if param.sem_mat
%             is defined. Diagonal entries are set to param.D.
%
%  semantic:  [list length X list length] matrix of scaled semantic
%             similarity values used to construct w_cf.

LL = size(pres_itemnos, 2);

n_start_units = 1;
n_item_units = LL;
n_ipi_units = 0;
n_ri_units = 0;

if isfield(param, 'B_ipi')  
  n_ipi_units = LL;
  env.ipi_dist_unit = [LL+1:(2*LL)];
end

if isfield(param, 'B_ri')  
  n_ri_units = 1;
  if isfield(param, 'B_ipi')  
    env.ri_dist_unit = (2*LL)+1;
  else
    env.ri_dist_unit = LL+1;
  end
end

n_units = n_start_units + n_item_units + n_ipi_units + n_ri_units;

env.s_unit = n_units;

% initialize the model representations
f = zeros(n_units, 1);
c = zeros(n_units, 1);
% this is the start unit
c(env.s_unit) = 1;

w_fc = eye(n_units) * (1 - param.G);
% w_fc = zeros(n_units);
w_cf = zeros(n_units);
w_cf_pre = zeros(n_units);

if param.Afc ~= 0
  % add constant connection strength between each item unit and
  % every other item unit
  w_fc(1:LL, 1:LL) = w_fc(1:LL, 1:LL) + param.Afc;
end

% modifying, this used to be w_cf, but then C can't project through
% f to f weights
if param.Acf ~= 0
  w_cf_pre(1:LL, 1:LL) = w_cf_pre(1:LL, 1:LL) + param.Acf;
end

if param.Dfc ~= 0
  % set diagonal strength (overrides C for diag entries), for
  % item self-associations
  for i = 1:LL
    w_fc(i,i) = param.Dfc;
  end
end

if param.Dcf ~= 0
  for i = 1:LL
    w_cf(i,i) = param.Dcf;
  end
end

if isfield(param, 'sem_mat') && ~isempty(param.sem_mat)
  % scale by a free parameter
  semantic = param.sem_mat(pres_itemnos, pres_itemnos);

  if param.Sfc ~= 0  
    w_fc(1:LL, 1:LL) = w_fc(1:LL, 1:LL) + semantic * param.Sfc;
  end
  if param.Scf ~= 0
    w_cf_pre(1:LL, 1:LL) = w_cf_pre(1:LL, 1:LL) + semantic * param.Scf;
  end
else
  semantic = [];
end
