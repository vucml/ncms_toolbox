% This function takes a word off the top of the word list, combines it with
% a drifted version of the context vector, then adds it to the trace stack.

function [newstack,newscramlist] = pres_word(param, oldstack, event, scramlist)

[height,~] = size(oldstack);

if height == 0 % if this is the first word, generate a random context vector
    contextarray = ones(1,param.n_cfeatures);
    for i = 1:length(contextarray)
        if rand >= param.AL
            contextarray(1,i) = -1;
        end
    end
else % if this isn't the first word, drift the previous word's context
    contextarray = oldstack(height,param.n_ifeatures + 1:param.n_ifeatures+param.n_cfeatures);
    for j = 1:length(contextarray)
        if rand <= param.PC * param.AL
           contextarray(1,j) = -contextarray(1,j); 
        end
    end
end

% grab an item off the top of the shuffled word list
itemarray = scramlist(1,:);
[n_words,~] = size(scramlist);
newscramlist = scramlist(2:n_words,:);

trace = horzcat(itemarray,contextarray);
newstack = [oldstack;trace];

end