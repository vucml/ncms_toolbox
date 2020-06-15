% Creates prototype vectors from which category members can be created
function prototypes = gen_prototypes(param)

prototypes = zeros(param.cpf * 4,param.n_ifeatures);

for i = 1:param.cpf * 4
    item = ones(1,param.n_ifeatures);
    for j = 1:param.n_ifeatures
        if rand >= param.AL
            item(1,j) = -1;
        end
    end
    prototypes(i,:) = item;
end

end