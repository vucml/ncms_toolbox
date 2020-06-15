% Creates the initial context array

function[init_context] = init_minerva(param)
% INPUTS: 
% param: structure containing parameters (AL, PC, n_ifeatures, n_cfeatures,
%        listlength).
%
% OUTPUTS:
% init_context: The initial context array.

init_context = ones(1,param.n_cfeatures);
for i = 1:param.n_cfeatures
    if rand >= param.AL
        init_context(1,i) = -1;
    end
end

end