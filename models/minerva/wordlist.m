% Creates a scrambled list of category words and filler words that
% pres_word plucks a word from.
function[scramarray,prototypes] = wordlist(param)

freq0 = zeros(1,param.cpf);
freq1 = ones(1,param.cpf);
freq3 = 3 * ones(1,param.cpf);
freq5 = 5 * ones(1,param.cpf);
freq_array = horzcat(freq0,freq1,freq3,freq5);

prototypes = gen_prototypes(param);
itemarray = zeros(9*param.cpf,param.n_ifeatures + 1);

index = 1;
for i = 1:param.cpf*4
    for j = 1:freq_array(1,i)
        item = prototypes(i,:);
        changeindexes = randsample(length(item),param.distance);
        for m = 1:length(changeindexes)
            item(1,changeindexes(m,1)) = -item(1,changeindexes(m,1));
        end
        item = [i,item];
        itemarray(index,:) = item;
        index = index + 1;
    end
end

scram = randperm(param.cpf*9);
scramarray = itemarray(scram,:);
scramarray = add_fillers(param,scramarray);

end