%
% Tutorial code for
% Polyn, S. M. Assessing neurocognitive hypotheses in a predictive
% model of the free-recall task.  

% set the seed for random number generation
% uncomment if you want it to always turn out the same way
% comment if you want it to vary from run to run
% rng(42)

% This tutorial provides a demonstration of predictive and
% generative modeling with the Context Maintenance and Retrieval (CMR)
% model 

%%
% Section 1.  Setting model parameters and task details
%%

% creating parameters for a basic version of CMR
param = struct();
% B_enc: context integration rate during encoding
param.B_enc = 0.7;
% B_rec: context integration rate during recall
param.B_rec = 0.5;
% P1 and P2 control the primacy mechanism (a learning rate boost
% for early serial positions)
param.P1 = 8;
param.P2 = 1;
% 'sampling_rule' and T control the input to the recall competition
% 'classic' means the version described in Howard & Kahana (2002) 
% (a form of softmax)
% T controls degree of non-linearity in the transformation from
% item support to probability of recall
param.sampling_rule = 'classic';
param.T = 0.35;
% 'stop_rule', X1, and X2 control likelihood of recall termination
% 'op': termination probability increases steadily with output position
param.stop_rule = 'op';
param.X1 = 0.001;
param.X2 = 0.5;
% Dfc and Dcf control the strength of pre-experimental associations
% (D refers to the diagonal elements of the associative matrices)
% fc / cf specifies direction of projection
param.Dfc = 3;
param.Dcf = 1;

% check_param_cmr is a helper script that runs through params and
% checks for missing fields 
param = check_param_cmr(param);

% Here we specify the free-recall task details 
% LL: list length
LL = 24;
n_trials = 120;
% since we are excluding repeats and intrusions, max_recalls == LL
max_recalls = LL;
% Here we create a data structure for our synthetic
% (model-generated) data
% each matrix attached to the data structure has a similar format
% each row corresponds to a different trial
data = struct();
% integer values indicate participant identity, here we pretend there
% is just one participant (since all trials are generated using the
% same parameters)
data.subject = ones(n_trials,1);
% the recalls matrix: each column corresponds to an output
% position, and the integer value corresponds to the serial
% position of the recalled item
data.recalls = zeros(n_trials,max_recalls);
% the presented item numbers matrix: each column corresponds to a
% serial position, with the integer value corresponding to the
% item's index in a word-pool.  Here we are not simulating item
% identities so we set these to dummy values of 1 to list length
data.pres_itemnos = repmat([1:LL],n_trials,1);

%%
% Section 2.  Generating synthetic data and plotting summary
% statistics from the data
%%

% use the CMR model to generate 120 trials of synthetic data
% (recall sequences)
seq = generate_cmr(param, data);
% store the recall sequences in the data structure
data.recalls = seq;

% figpath determines where on disk the code will save any created
% figures, if figpath is an empty string, the code will not save
% the figures to disk 

%figpath = '';
figpath = '~/Science/GitHub/chapter-predictive-cmr/figs/';

% calculate and plot some basic summary statistics
% these functions are in the EMBAM toolbox from GitHub
% (episodic memory behavioral analysis toolbox)
% github.com/vucml/EMBAM

% serial position curve
figure(1); clf;
% spc function returns a matrix of recall probabilities
% plot_spc function makes the graph
plot_spc(spc(seq,data.subject,LL));
if ~isempty(figpath)
  print('-depsc', fullfile(figpath,'basic_spc.eps'));
end

% lag-based conditional response probability analysis (lag-CRP)
figure(2); clf;
% crp function returns a matrix of transition probabilities
% plot_crp function makes the graph
plot_crp(crp(seq,data.subject,LL));
if ~isempty(figpath)
  print('-depsc', fullfile(figpath,'basic_crp.eps'));
end

%%
% Section 3. Using predictive simulations to perform parameter recovery
%%

% A simple test of parameter recovery
% create 11 model variants with different values of 
% beta rec, which controls temporal reinstatement
% evaluate likelihood of the synthetic data for each model variant
fstruct.data = data;
fstruct.verbose = false;
B_rec_vals = [0:0.1:1];
for i=1:length(B_rec_vals)
  temp_param = param;
  temp_param.B_rec = B_rec_vals(i);
  err(i) = eval_param_cmr(temp_param, fstruct);
end

% demonstrate best-fitting value matching the generating value
figure(3); clf;
plot(B_rec_vals,-err,'ko-');
hold on;

xlabel('\beta_{rec} value');
ylabel('log-likelihood');
h = gca;
% grab the current y-axis range
ylim = h.YLim;
% add a line representing the generating value of beta rec
plot([param.B_rec param.B_rec],[h.YLim(1) h.YLim(2)],'r--');
h.Children(1).LineWidth = 1.5;
h.Children(2).LineWidth = 1.5;
h.Children(2).MarkerFaceColor = 'w';
h.Children(2).MarkerSize = 8;
h.FontSize=14;

if ~isempty(figpath)
  print('-depsc', fullfile(figpath,'B_rec_sweep.eps'));
end

%%
% Section 4. Creating synthetic neural data and linking it to the
% temporal context reinstatement process.  Generating synthetic
% behavioral data using this 'neural' model. 
%%

% construct variable beta rec model
% beta rec centers on 0.5 with random normal deviations
% truncated at 0 and 1 
base_B_rec = 0.5;
% can alter this
neural_signal_strength = 0.2;
synth_neural_fluct = randn(n_trials, max_recalls) * neural_signal_strength;
var_B_rec = base_B_rec + synth_neural_fluct;
var_B_rec(var_B_rec<0) = 0;
var_B_rec(var_B_rec>1) = 1;

% make variable parameter matrix
% beta rec will vary from recall event to recall event
orig_var_param = struct();
orig_var_param(1).name = 'B_rec';
orig_var_param(1).update_level = 'recall_event';
orig_var_param(1).val = var_B_rec;
% this creates a more efficient structure for controlling parameter fluctuations
% trials x events x parameters
orig_vp_mat = create_param_mat(orig_var_param, n_trials, LL, max_recalls);
% synthetic data with variability in beta rec
seq = generate_cmr(param, data, orig_vp_mat);
% make a copy of the data struct from above
var_data = data;
% put these newly generated recall events on the data struct
var_data.recalls = seq;

% demonstration of how the lag-CRP changes for low vs high levels
% of temporal reinstatement

% create masks for recall events with low vs high temporal reinstatement
lo_B_rec_mask = var_B_rec < 0.5;
hi_B_rec_mask = var_B_rec > 0.5;

% lag-CRP using a 'from_rec_mask' 
% each recall transition is from one item and to another, this mask
% is applied on the 'from' side
%lo_lc = crp(var_data.recalls, ...
%            var_data.subject, ...
%            LL, lo_B_rec_mask);

%hi_lc = crp(var_data.recalls, ...
%            var_data.subject, ...
%            LL, hi_B_rec_mask);

lo_lc = crp(var_data.recalls, ...
            var_data.subject, ...
            LL, lo_B_rec_mask, ...
            make_clean_recalls_mask2d(var_data.recalls));

hi_lc = crp(var_data.recalls, ...
            var_data.subject, ...
            LL, hi_B_rec_mask, ...
            make_clean_recalls_mask2d(var_data.recalls));

figure(4); clf;
plot([-5:5],lo_lc(:,LL-5:LL+5),'ko-')
hold on;
plot([-5:5],hi_lc(:,LL-5:LL+5),'r^-')

% make the figure look pretty
h = gca();
h.Children(1).MarkerSize = 10;
h.Children(1).MarkerFaceColor = 'w';
h.Children(1).LineWidth = 1.5;
h.Children(2).MarkerSize = 10;
h.Children(2).MarkerFaceColor = 'w';
h.Children(2).LineWidth = 1.5;
h.FontSize = 16;
h.XLim = [-5.5 5.5];
h.YLim = [0 0.5];
h.XTick = [-5 -4 -3 -2 -1 0 1 2 3 4 5];
h.YTick = [0:0.1:0.5];
h.XLabel.String = 'Lag';
h.YLabel.String = 'Conditional Response Prob.';

if ~isempty(figpath)
  print('-depsc', fullfile(figpath,'var_crp.eps'));
end


%%
% Section 5. Predictive simulations given the data from the model
% with variable temporal reinstatement.
%%

% what's the likelihood under the perfect case where B_rec
% fluctuations perfectly match those used to generate the data
fstruct = struct();
fstruct.data = var_data;
fstruct.var_param = orig_vp_mat;
fstruct.verbose = false;
best_case_err = eval_param_cmr(temp_param, fstruct);

% turning variable B rec into synthetic neural data

% adding noise before normalizing
noise_strength = 0.1;
temp_neural = synth_neural_fluct;
temp_neural = temp_neural + (randn(n_trials, max_recalls) * noise_strength);

neural_mat = zeros(n_trials, max_recalls);
for i=1:n_trials
  neural_mat(i,:) = (temp_neural(i,:) - mean(temp_neural(i,:))) / std(temp_neural(i,:));
end
% add a bit of noise to see how robust it is
% reliably prefers the neural version even when noise is as strong
% as the original neural signal fluctuations
% when noise_strength is 0.6 it seems to fail about half the time
%noise_strength = 0.1;
%neural_mat = neural_mat + (randn(n_trials, max_recalls) * noise_strength);

fstruct = struct();
fstruct.verbose = false;
fstruct.data = var_data;

% Grid search
% Get likelihood for different values of beta rec and neural
% scaling parameter

B_rec_vals = [0:0.1:1];
nsp_vals = [.1 .2];
% param_grid(1,:,:) gives the B_rec vals, dim 3 
% param_grid(2,:,:) gives the neural scaling vals, dim 2
param_grid = zeros(2,length(nsp_vals),length(B_rec_vals));
[param_grid_B_rec param_grid_nsp] = meshgrid(B_rec_vals, nsp_vals);
param_sweep_B_rec = reshape(param_grid_B_rec,[],1);
param_sweep_nsp = reshape(param_grid_nsp,[],1);
LL_naive = zeros(1,length(B_rec_vals));
LL_sweep_neural = zeros(length(param_sweep_B_rec),1);

% log likelihood for neurally naive model variants
for i=1:length(B_rec_vals)
  temp_param = param;
  temp_param.B_rec = B_rec_vals(i);
  temp_param.n = 0;
  out = demo_wrap_eval(temp_param, neural_mat, fstruct);
  LL_naive(i) = -out.err;
end

% log likelihood for neurally informed model variants
% if we reshape the meshgrid to be a column, can get the LLs and
% the params to all be columns which makes bookkeeping easier below
for i=1:length(param_sweep_B_rec)
  
  temp_param = param;
  %temp_param.B_rec = B_rec_vals(i);
  temp_param.B_rec = param_sweep_B_rec(i);
  %temp_param.n = nsp_vals(j);
  temp_param.n = param_sweep_nsp(i);
  
  out = demo_wrap_eval(temp_param, neural_mat, fstruct);    
  %err = out.err;
  LL_sweep_neural(i) = -out.err;
  %tr_err(i,j) = err;
  
end

% reshaping here is just to make graphing easier
LL_grid_neural = reshape(LL_sweep_neural,length(nsp_vals),length(B_rec_vals));

figure(5); clf;
% neurally naive
plot([0:0.1:1],LL_naive(1,:),'Color',[0 0 1],'Marker','d');
hold on;
% neural scaling 0.1
plot([0:0.1:1],LL_grid_neural(1,:),'Color',[1 0 0],'Marker','o');
% neural scaling 0.2
plot([0:0.1:1],LL_grid_neural(2,:),'Color',[0.5 0 0.5],'Marker','^');
xlabel('\beta_{rec} value');
ylabel('log-likelihood');
h = gca();
h.YLim = [-3000 -2700];
plot([param.B_rec param.B_rec],[h.YLim(1) h.YLim(2)],'r--');
for i=2:4
  h.Children(i).LineWidth = 1.5;
  %h.Children(i).Marker = 'o';
  h.Children(i).MarkerSize = 8;
  h.Children(i).MarkerFaceColor = 'w';
end
h.FontSize=14;

legend('\nu = 0','\nu = 0.1', '\nu = 0.2', 'Location', 'SouthEast');
if ~isempty(figpath)
  print('-depsc', fullfile(figpath,'B_nu_sweep.eps'));
end

%%
% Section 6.  Model comparison statistics.
%%


% AIC with correction for finite samples
% n is number of estimated data points
% V is number of free param
% L is log-likelihood
% 2*L + 2*V + (2*V*(V+1)) / (n-V-1)

% count up the number of recall events, this is the number of data points
% we add n_trials because the model counts recall termination as a
% data point; it tries to predict this just like it tries to
% predict identity of recalled items
ndata = sum(fstruct.data.recalls>0,'all') + n_trials;

% here only B_rec is free
V = 1;
for i=1:length(LL_naive)
  AIC_naive(i) = -2*LL_naive(i) + 2*V + ((2*V*(V+1)) / (ndata-V-1));
end

% B_rec and neural scaling are free
V = 2;
for i=1:length(LL_sweep_neural)
  AIC_sweep_neural(i) = -2*LL_sweep_neural(i) + 2*V + ((2*V*(V+1)) / (ndata-V-1));
end

% find the best score
% let's say we just want to compare the best fitting model where 
% neural scaling was zero, vs best where neural scaling was greater
% than zero
maxl=zeros(1,2);

% naive
[maxl(1) ind_naive] = min(AIC_naive);

fprintf('\nNeurally naive max-likelihood\n');
fprintf('Max-likelihood beta_rec: %.2f\n', B_rec_vals(ind_naive));
fprintf('Likelihood for neurally naive model: %.2f\n', LL_naive(ind_naive));
fprintf('AIC score, neurally naive: %.2f\n', AIC_naive(ind_naive));

% neural scaling on
[maxl(2) ind_neural] = min(AIC_sweep_neural);

% compare naive vs neural
% smaller logl is better
% this becomes the reference score
minval = min(maxl);

% weighted AIC 
% in demo example, this should show that neurally informed model is 
% vastly preferred to the neurally naive model
temp_aic = exp(-0.5*(maxl-minval));
wAIC = temp_aic ./ sum(temp_aic);

fprintf('\nNeurally informed max-likelihood\n');
fprintf('Max-likelihood beta_rec: %.2f\n', param_sweep_B_rec(ind_neural));
fprintf('Max-likelihood neural scaling: %.2f\n', param_sweep_nsp(ind_neural));
fprintf('Likelihood for neurally informed model: %.2f\n', LL_sweep_neural(ind_neural));
fprintf('AIC score, neurally informed: %.2f\n', AIC_sweep_neural(ind_neural));
fprintf('\nComparison of neurally naive and neurally informed models\n');
fprintf('weighted AIC for neurally naive model: %f\n', wAIC(1));
fprintf('weighted AIC for neurally informed model: %f\n', wAIC(2));

%%
% Section 7.  Some final tests
%%

% what happens if you scramble the neural signal by permuting the
% rows, this is an alternative baseline

% set n_scrambles to 100 for a more accurate p-value (though the code will
% take several seconds longer to run)
n_scrambles = 20;
LL_permtest = zeros(1, n_scrambles);

% for this exploration of permutation statistics we can assume we
% know the true generating parameters of B_rec and n, then we can
% see how scrambling the neural signal affects the likelihood
% scores under otherwise perfect conditions

for i = 1:n_scrambles
  
  temp_param = param;

  temp_param.B_rec = base_B_rec;
  temp_param.n = neural_signal_strength;
  
  row_perm = randperm(n_trials);
  scram_neural_mat = neural_mat(row_perm,:);
  
  out = demo_wrap_eval(temp_param, scram_neural_mat, fstruct);    

  LL_permtest(i) = -out.err;
  
end

% we can get a p-value out of this permutation analysis by
% comparing the log-likelihood of the model with the unscrambled
% neural signal, to the distribution of log-likelihoods with
% different scrambles of the neural signal

pval = sum(LL_permtest>LL_sweep_neural(ind_neural))/n_scrambles;

fprintf('\nBest log-likelihood with scrambled neural signal: %.2f\n',min(LL_permtest));
fprintf('p-value for neurally informed model against permut. distrib.: %f\n',pval);




