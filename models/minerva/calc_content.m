% Creates the primary echo

function[echo] = calc_content(param,activations,memstack)
% INPUTS: 
%       param: structure containing parameters.
% 
% activations: array of length param.listlength containing an activation level
%              for each study item.
%    
%    memstack: [listlength X n_ifeatures + n_cfeatures] matrix containing
%              memory traces for all study items.
%
% OUTPUTS:
% echo: array of length param.n_ifeatures + param.n_cfeatures where each
%       feature is the sum of corresponding trace features weighted by trace
%       activations.
echo = zeros(1,param.n_ifeatures + param.n_cfeatures);

for c = 1:length(echo)
    sum = 0;
    for r = 1:param.listlength
        sum = sum + activations(1,r) * memstack(r,c);
    end
    echo(1,c) = sum;
end