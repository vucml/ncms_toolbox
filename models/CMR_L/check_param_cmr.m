function param = check_param_cmr(param)
%CHECK_PARAM_CMR   Prepare parameters for running CMR.
%
%  param = check_param_cmr(param)

if ~isfield(param, 'lat_inh')
  param.lat_inh = 0;
end

if ~isfield(param, 'B_s_init_transient')
  param.B_s_init_transient = false;
end


if isfield(param, 'B')
  % convert to two-parameter version
  if ~isfield(param, 'B_enc')
    param.B_enc = param.B;
  end
  if ~isfield(param, 'B_rec')
    param.B_rec = param.B;
  end
end

if isfield(param, 'P')
  % convert to two-parameter version
  if ~isfield(param, 'P1')
    param.P1 = param.P;
  end
end

if ~isfield(param, 'Dfc')
  if isfield(param, 'G') && ~isempty(param.G)
    param.Dfc = (1 - param.G) / param.G;
    param.G = 1;
  else
    param.Dfc = 0;
  end
else
  % setting diagonal, so don't need the G parameter to vary
  param.G = 1;
end
if isfield(param, 'D')
  param.Dcf = param.D;
elseif ~isfield(param, 'Dcf')
  param.Dcf = 0;
end

if ~isfield(param, 'P2')
  % default is that only serial position 1 is affected by primacy boost
  param.P2 = 1000000;
end

if ~isfield(param, 'Lfc')
  param.Lfc = 1;
end

if ~isfield(param, 'Lcf')
  param.Lcf = 1;
end

% semantic related
% if C exists it should override Afc
% which means you have to be careful
% if C isn't defined don't define it
if isfield(param, 'C')
  param.Acf = param.C;
else
  % if C isn't defined check if Acf is defined
  if ~isfield(param, 'Acf')
    param.Acf = 0;
  end
end
  
if isfield(param, 'S')
  param.Scf = param.S;
end

if ~isfield(param, 'Sfc')
  param.Sfc = 0;
end

if ~isfield(param, 'Afc')
  param.Afc = 0;
end
  
% if ~isfield(param, 'Acf')
%   param.Acf = param.C;
% end
% if isfield(param, 'C')
%   param.Acf = param.C;
% elseif ~isfield(param, 'Acf')
%   param.Acf = 0;
% end
% if ~isfield(param, 'Afc')
%   param.Afc = 0;
% end

if ~isfield(param, 'T')
  param.T = 1;
end

if ~isfield(param, 'stop_rule')
  param.stop_rule = 'op';
end

if ~isfield(param, 'sampling_rule')
  param.sampling_rule = 'classic';
end
