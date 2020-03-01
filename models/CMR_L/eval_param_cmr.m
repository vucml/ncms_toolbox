function [err, logl, logl_all] = eval_param_cmr(param, varargin)
%EVAL_PARAM_CMR   Calculate likelihood for CMR with a given set of paramters.
%
%  [err, logl, logl_all] = eval_param_cmr(param, ...)
%
%  INPUTS:
%     param:  parameter structure, or numeric vector of parameter
%             values (if numeric, must also pass param_info; see below).
%
%  OUTPUTS:
%      logl:  likelihood of the observed data.
%
%  logl_all:  likelihood for possible outcome, conditional on the
%             recalls made up to that point in the observed data.
%
%  OPTIONS:
%  These options may be set using parameter, value pairs, or by
%  passing a structure with these fields. Defaults shown in parentheses.
%   data          - REQUIRED. Either a behavioral data structure,
%                   or the path to a MAT-file containing the data,
%                   saved as a variable named 'data'.
%   param_info    - see unpack_param for details.
%   f_logl        - handle to a function of the form:
%                    [logl, logl_all] = f_logl(param, data)
%                   Calculates likelihood. (@cmr_general)
%   f_check_param - handle to a function of the form:
%                    param = f_check_param(param)
%                   Used to set default values and run sanity
%                   checks on parameters. (@check_param_cmr)
%   verbose       - if true, more information is printed.
%                   (isstruct(param))
%  May also pass additional parameter fields for f_logl.

% param evaluation configuration
def.data = '';
def.param_info = [];
def.var_param = [];
def.f_logl = @cmr_general;
def.f_check_param = @check_param_cmr;
def.verbose = isstruct(param);
def.load_data = true;
[opt, custom_param] = propval_lite(varargin, def);

if ~isstruct(param)
  % convert to struct format
  if isempty(opt.param_info)
    error('Cannot interpret parameter vector without param_info')
  end
  param = unpack_param(param, opt.param_info);
end

% merge in additional parameters set
if ~isempty(custom_param)
  param = propval_lite(custom_param, param);
end

% sanity checks, set default parameters, etc.
param = opt.f_check_param(param);

if opt.verbose
  disp(param)
end

% load the behavioral data if necessary
if opt.load_data && ischar(opt.data)
  opt.data = getfield(load(opt.data, 'data'), 'data');
end

% calculate log likelihood
%tic
if nargout(opt.f_logl) == 2
  [logl, logl_all] = opt.f_logl(param, opt.data, opt.var_param);
else
  logl = opt.f_logl(param, opt.data, opt.var_param);
  logl_all = [];
end
err = -nansum(logl(:));

if opt.verbose
  %fprintf('Log likelihood: %.4f\n%.3f seconds elapsed.\n', -err, toc)
  fprintf('Log likelihood: %.4f\n', -err)
end

