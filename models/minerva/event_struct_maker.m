% This function creates an array of event structures.

function [events] = event_struct_maker(param)

sess_start = struct('type','SESS_START');

for i = 1 : 9*param.cpf
    pres_word(i) = struct('type','PRES_WORD');
end

for j = 1 : 4*param.cpf
    pres_studied_word(j) = struct('type','PRES_STUDIED_WORD');
end

for k = 1 : 4*param.cpf
    pres_lure(k) = struct('type','PRES_LURE');
end

pres = horzcat(pres_studied_word,pres_lure);
pres = pres(randperm(8*param.cpf));
events = [sess_start,pres_word,pres];

end