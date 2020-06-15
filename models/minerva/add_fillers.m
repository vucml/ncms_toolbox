% Adds filler words (category 0) to a word list, breaking up repeats.
function [newnewlist] = add_fillers(param,list)

n_fillers = 7;
fillers = ones(n_fillers,param.n_ifeatures+1);
for h = 1:n_fillers
   fillers(h,1) = 0;
   for g = 2:param.n_ifeatures+1
       if rand <= param.AL
           fillers(h,g) = -1;
       end
   end
end

newlist(1,:) = list(1,:);

[LL,~] = size(list);
count = 2;
fillercount = 1;
for i = 2:LL
    if list(i,1) == list(i-1,1)
        newlist(count,:) = fillers(fillercount,:);
        count = count + 1;
        fillercount = fillercount + 1;
        newlist(count,:) = list(i,:);
        count = count + 1;
    else
        newlist(count,:) = list(i,:);
        count = count + 1;
    end
end

fillindex = randperm(LL+n_fillers,n_fillers-(fillercount-1));
mask = zeros(1,LL+n_fillers); % probably could've used a mask for breaking up repeats, too
for j = 1:length(fillindex)
    mask(fillindex(j)) = 1;
end
count = 1;
newnewlist = zeros(LL+n_fillers,param.n_ifeatures + 1);
for k = 1:length(mask)
    if mask(k) == 0;
        newnewlist(k,:) = newlist(count,:);
        count = count + 1;
    else
        newnewlist(k,:) = fillers(fillercount,:);
        fillercount = fillercount + 1;
    end
end