% Calculates the activation level for each study item

function[activations] = calc_activations(param,probe,memstack)
% INPUTS: 
%    param: structure containing parameters (AL, PC, n_ifeatures, n_cfeatures,
%           listlength).
%    
%    probe: array that acts as a "stimulus." For free recall, the context
%           elements of the probe are filled in, while the item elements
%           are set
%           to 0.
% 
% memstack: [listlength X n_ifeatures + n_cfeatures] matrix containing
%           memory traces for all study items.
%
% OUTPUTS:
% activations: array of length listlength containing an activation level
%              for each study item.

activations = zeros(1,param.listlength);

for i = 1:param.listlength
    similarity = 0;
    for j = 1:length(probe)
        similarity = similarity + (probe(1,j) * memstack(i,j)) / ...
            (param.n_ifeatures + param.n_cfeatures);
    end
    activations(1,i) = similarity ^ 3;
end

end