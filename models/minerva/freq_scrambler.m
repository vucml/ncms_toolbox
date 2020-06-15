function[itemarray] = freq_scrambler(param)

freq0 = zeros(1,param.cpf);
freq1 = ones(1,param.cpf);
freq3 = 3 * ones(1,param.cpf);
freq5 = 5 * ones(1,param.cpf);

freq_array = horzcat(freq0,freq1,freq3,freq5);
scram_indexes = randperm(4 * cpf);
scram_freq_array = freq_array(scram_indexes);

prototypes = gen_prototypes(param);
itemarray = zeros(9*param.cpf,param.n_ifeatures);

index = 1;
for i = 1:param.cpf*4
    for j = 1:scram_freq_array(1,i)
        item = prototypes(i,:);
        changeindexes = randsample(length(item),param.distance);
        for m = 1:length(changeindexes)
            item(1,changeindexes(m,1)) = -item(1,changeindexes(m,1));
        end
        itemarray(index,:) = item;
        index = index + 1;
    end
end