% This script initializes the param structure and strings together all of
% the functions so you don't have to manually run each function in order
% every time you want to test a new function.

% Well, right now all it's doing is initializing the param structure.
param = struct();
param.AL = 0.5;
param.PC = 0.2;
param.n_ifeatures = 10;
param.n_cfeatures = 10;
param.listlength = 24;
param.cpf = 1; % categories per feature
param.distance = 3;

% events = event_struct_maker(param);
% memstack = simulate_minerva(param,events);

%init_context = init_minerva(param);
%memstack = create_stack(init_context,param);
%activations = calc_activations(param,[0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],memstack);
%intensity = calc_intensity(activations);
%echo = calc_content(param,activations,memstack);

%n_subjects = 500;
%subjects = ones(1,n_subjects);
%recallmatrix = zeros(n_subjects,param.listlength);
%weightmatrix = zeros(param.listlength,param.listlength,n_subjects);
%for i = 1:n_subjects
    %init_context = init_minerva(param);
    %memstack = create_stack(init_context,param);
    %[recallmatrix(i,:),weightmatrix(:,:,i)] = free_recall(memstack,param);
%end

%stop_probs = stop_indexes(recallmatrix);

%lagcrp = crp(recallmatrix,subjects,param.listlength);
%plot(lagcrp)
%print('~/Documents/results/minerva/lag_0.2_3.eps', '-depsc')