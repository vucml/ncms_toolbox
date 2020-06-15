% Creates a histogram of indexes at which free_recall terminated recall. 

function [stop_probs] = stop_indexes(recallmatrix)
% INPUTS:
% recallmatrix: [n_subjects X param.listlength] matrix of recalls for
%               several subjects
%
% OUTPUTS:
% stops: array of length n_subjects containing the stop index for each
%        subject

[rows,cols] = size(recallmatrix);
stop_hist = zeros(1,cols+1);
stop_probs = zeros(1,cols+1);

for r = 1:rows
    for c = 1:cols
        if recallmatrix(r,c) == 0
            stop_hist(1,c) = stop_hist(1,c) + 1;
            break
        elseif c == cols
            stop_hist(1,c+1) = stop_hist(1,c+1) + 1;
        end    
    end
end

plot(stop_hist)
print('~/Documents/results/minerva/stop_plot', '-depsc')

for i = 1:cols+1
    stop_probs(1,i) = stop_hist(1,i)/sum(stop_hist);
end

end