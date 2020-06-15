function[logl] = calc_log_likelihood(recallmatrix,weightmatrix,stop_probs)

[n_subjects,listlength] = size(recallmatrix);
logl = ones(1,n_subjects);

% Converting the weights in weightmatrix into probabilities
for p = 1:n_subjects
    for r = 1:listlength
        weightsum = sum(weightmatrix(r,:,p));
        for c = 1:listlength
            if weightsum > 0
                weightmatrix(r,c,p) = weightmatrix(r,c,p) / weightsum;
            else
                weightmatrix(r,c,p) = 0;
            end
        end
    end
end

for i = 1:n_subjects
    for j = 1:listlength
        if recallmatrix(i,j) == 0
            logl(1,i) = logl(1,i) * stop_probs(1,j);
            break;
        else
            logl(1,i) = logl(1,i) * weightmatrix(j,recallmatrix(i,j),i);
        end
    end
end

logl = log10(logl);

end