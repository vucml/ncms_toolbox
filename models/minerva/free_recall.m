% Generates a free recall sequence (including stop probability).

function[recall_seq,weightmatrix] = free_recall(memstack,param)
% INPUTS:
% memstack: [listlength X n_ifeatures + n_cfeatures] matrix containing
%           memory traces for all study items.
%
%    param: structure containing parameters (AL, PC, n_ifeatures,
%               n_cfeatures, listlength).
%
% OUTPUTS:
% recall_seq: array of length param.listlength containing index numbers of
%             recalled items. 0's indicate that recall has terminated.

% making a probe by drifting the last trace and setting item features to 0
probearray = memstack(param.listlength,:);
for i = 1:param.n_ifeatures
    probearray(1,i) = 0;
end
for j = param.n_ifeatures + 1: param.n_cfeatures
    if rand < param.PC * param.AL
        probearray(1,j) = -probearray(1,j);
    end
end

% calculating activations for the initial probe and converting them into
% weights
weightmatrix = zeros(param.listlength);
activations = calc_activations(param,probearray,memstack);
smallest_act = 1;
    for p = 1:param.listlength
        if activations(1,p) < smallest_act
            smallest_act = activations(1,p);
        end
    end
weights = sqrt(activations - smallest_act);

recall_seq = zeros(1,param.listlength);

intensities = zeros(1,param.listlength);
recalled = zeros(param.listlength,param.n_ifeatures + param.n_cfeatures);

for k = 1:param.listlength
    weightmatrix(k,:) = weights;
    
    % RECALLING A STUDY ITEM
    population = 1:param.listlength;
    % weighted sampling of study items
    if any(weights)
        recall_seq(1,k) = randsample(population,1,true,weights);
    else % in case all the weights happen to be zero, randomly sample from remaining items
        for r = 1:k-1
            population(recall_seq(1,r)) = 0;
            population(population == 0) = [];
        end    
        recall_seq(1,k) = datasample(population,1);
    end
    
    % DECIDING WHETHER TO TERMINATE RECALL
    recalled(k,:) = memstack(recall_seq(1,k),:);
    % setting recalled item as probe and recalculating activations
    activations = calc_activations(param,memstack(recall_seq(1,k),:),memstack);
    % calculating intensity for not yet recalled items
    rec_activations = calc_activations(param,memstack(recall_seq(1,k),:),recalled);
    intensities(1,k) = calc_intensity(activations) - calc_intensity(rec_activations);
    % stop probability thing
    if intensities(1,k) < 0.5
        break;
    end
    
    % RECALCULATING WEIGHTS
    smallest_act = 1;
    for n = 1:param.listlength
        if activations(1,n) < smallest_act
            smallest_act = activations(1,n);
        end
    end
    weights = sqrt(activations - smallest_act);
    % setting weights for already recalled items to 0 (to avoid repetition)
    for m = 1:k
        weights(1,recall_seq(1,m)) = 0;
    end
end

end
        