% Creates stack of memory traces (presented item array + context array)

function[memstack] = create_stack(init_context,param)
% INPUTS:
% init_context: Initial context. The output of the init_minerva function.
%
%        param: structure containing parameters (AL, PC, n_ifeatures,
%               n_cfeatures, listlength).
%
% OUTPUTS:
% memstack: [listlength X n_ifeatures + n_cfeatures] matrix containing
%           memory traces for all study items.

itemstack = ones(param.listlength,param.n_ifeatures);
contextstack = ones(param.listlength,param.n_cfeatures);

changerate = 0.2;

for i = 1:param.listlength
    newitem = ones(1,param.n_ifeatures);
    for j = 1:param.n_ifeatures
        if rand >= param.AL
            newitem(1,j) = -1;
        end
    end
    if i == 1
        itemstack(i,:) = newitem;
    else
        itemstack(i,:) = itemstack(i-1,:);
        for k = 1:param.n_ifeatures
            if rand <= changerate
                itemstack(i,k) = newitem(1,k);
            end
        end
    end
end
                
contextstack(1,:) = init_context;
for m = 2:param.listlength
    for n = 1:param.n_cfeatures
        if rand <= param.PC * param.AL
            init_context(1,n) = -init_context(1,n);
        end
    end
    contextstack(m,:) = init_context;
end

memstack = horzcat(itemstack,contextstack);

end           