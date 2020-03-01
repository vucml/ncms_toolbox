function [fstruct, ranges] = setup_search_cmr(search, opt_fin)
%SETUP_SEARCH_CMR
%
% Generic search setup function.  Creates the fstruct used during
% parameter searches and returns allowable ranges on the parameters

% behavioral data
data_path = fullfile(opt_fin.res_dir, opt_fin.data_file);
data = getfield(load(data_path, 'data'), 'data');
data.recalls = clean_recalls(data.recalls);

[param_info, fixed] = search_param_cmr(search);

fstruct = fixed;
fstruct.param_info = param_info;
fstruct.f_logl = @cmr_general;
fstruct.data = data;
fstruct.n_rep = 1;
fstruct.load_data = false;

% update or add in any specialty parameters
if exist('opt_fin','var')
  fstruct = propval(opt_fin, fstruct, 'strict', false);
end

ranges = cat(1, param_info.range);

% set random seeds (very important)
rng('shuffle');
