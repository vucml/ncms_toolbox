function [param_info, fixed] = search_param_cmr(model_type, split_names, ...
                                                n_groups)
%SEARCH_PARAM_CMR   Get parameter ranges and fixed parameters.
%
%  [param_info, fixed] = search_param_cmr(model_type, split_names, n_groups)

fixed = struct;
switch model_type
  case 'recovery1'

    par.B_enc = [0 1];
    par.B_rec = [0 1];
    
    fixed.P1 = 0;
    fixed.P2 = 0;
    fixed.G = 0.5;
    fixed.X1 = 0.01;
    fixed.X2 = 0.2;
    fixed.C = 0;
    fixed.T = 1;
    fixed.stop_rule = 'op';
    fixed.sampling_rule = 'classic';

  
  case 'sample'
    par.B = [0 1];
    par.P1 = [0 10];
    par.P2 = [0 10];
    par.X1 = [0 .1];
    par.X2 = [0 1];
    par.C = [0 1];
    par.G = [0 1];
    
    start.B = 0.5;
    start.P1 = 1;
    start.P2 = 1;
    start.X1 = 0.0001;
    start.X2 = 0.3;
    start.C = 0.1;
    start.G = 0.9;
    
    fixed.T = 1;

  otherwise
    error('Unknown model type: %s', model_type)
end

names = fieldnames(par);
ranges = struct2cell(par);
if exist('start', 'var')
  starts = struct2cell(start);
else
  starts = [];
end

if exist('n_groups', 'var')
  temp = ones(size(names));
  for i = 1:length(split_names)
    ind = find(strcmp(split_names{i}, names));
    temp(ind) = n_groups;
  end
  n_groups = temp;
else
  n_groups = [];
end

param_info = make_param_info(names, 'range', ranges, 'start', starts, ...
                             'n_groups', n_groups);


