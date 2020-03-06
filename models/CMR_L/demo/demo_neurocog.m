
% set the seed for rng
% uncomment if you want it to always turn out the same way
rng(42)

% demonstration of predictive and generative modeling with CMR

% set parameters for a basic version of CMR
param = struct();
% context integration rate during encoding
param.B_enc = 0.7;
% context integration rate during recall
param.B_rec = 0.5;
% primacy learning rate boost
param.P1 = 8;
param.P2 = 1;
% non-linearity in the recall competition 
% 'sampling rule' controls calculation of item support in the competition
param.sampling_rule = 'classic';
param.T = 0.35;
% likelihood of recall termination
param.stop_rule = 'op';
param.X1 = 0.001;
param.X2 = 0.6;
% the strength of pre-experimental associations
% D means the diagonal elements of the associative matrices
% fc / cf specifies direction of projection
param.Dfc = 3;
param.Dcf = 1;

% runs through params and checks for missing fields
param = check_param_cmr(param);

% some task details 
LL = 24;
n_trials = 120;
max_recalls = LL;
% creating a data structure for our synthetic (model generated) data
data = struct();
data.subject = ones(n_trials,1);
data.recalls = zeros(n_trials,max_recalls);
data.pres_itemnos = repmat([1:LL],n_trials,1);

% use the CMR model to generate 120 trials of synthetic data
seq = generate_cmr(param, data);
% store the recall sequences in the data structure
data.recalls = seq;

%figpath = '';
figpath = '~/Science/GitHub/chapter-predictive-cmr/figs/';

% calculate and plot some basic summary statistics
% uses EMBAM functions
figure(1); clf;
plot_spc(spc(seq,data.subject,LL));
if ~isempty(figpath)
  print('-depsc', fullfile(figpath,'basic_spc.eps'));
  system('epstopdf basic_spc.eps');
end
figure(2); clf;
plot_crp(crp(seq,data.subject,LL));
if ~isempty(figpath)
  print('-depsc', fullfile(figpath,'basic_crp.eps'));
  system('epstopdf basic_crp.eps');
end


% a simple test of parameter recovery
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
plot(B_rec_vals,err,'ko-');
hold on;

xlabel('\beta_{rec} value');
ylabel('|log-likelihood|');
h = gca;
% grab the current y-axis range
ylim = h.YLim;
% add a line representing the generating value of beta rec
plot([param.B_rec param.B_rec],[h.YLim(1) h.YLim(2)],'r--');
h.Children(1).LineWidth = 2;
h.Children(1).MarkerFaceColor = 'w';
h.FontSize=14;


% construct variable beta rec model
% beta rec centers on 0.5 with random normal deviations
% truncated at 0 and 1 
base_B_rec = 0.5;
% can alter this
neural_signal_strength = 0.1;
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

% what's the likelihood under the perfect case where B_rec
% fluctuations perfectly match those used to generate the data
fstruct = struct();
fstruct.data = var_data;
fstruct.var_param = orig_vp_mat;
fstruct.verbose = false;
best_case_err = eval_param_cmr(temp_param, fstruct);

% turning variable B rec into synthetic neural data
neural_mat = zeros(n_trials, max_recalls);
for i=1:n_trials
  neural_mat(i,:) = (synth_neural_fluct(i,:) - mean(synth_neural_fluct(i,:))) / std(synth_neural_fluct(i,:));
end
% add a bit of noise to see how robust it is
% reliably prefers the neural version even when noise is as strong
% as the original neural signal fluctuations
% when noise_strength is 0.6 it seems to fail about half the time
noise_strength = 0.1;
neural_mat = neural_mat + (randn(n_trials, max_recalls) * noise_strength);

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
  LL_naive(i) = out.err;
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
  LL_sweep_neural(i) = out.err;
  %tr_err(i,j) = err;
  
end

% reshaping here is just to make graphing easier
LL_grid_neural = reshape(LL_sweep_neural,length(nsp_vals),length(B_rec_vals));

figure(4); clf;
% neurally naive
plot([0:0.1:1],LL_naive(1,:),'go-');
hold on;
% neural scaling 0.1

plot([0:0.1:1],LL_grid_neural(1,:),'bo-');
% neural scaling 0.2
plot([0:0.1:1],LL_grid_neural(2,:),'ro-');
xlabel('\beta_{rec} value');
ylabel('|log-likelihood|');
h = gca();
plot([param.B_rec param.B_rec],[h.YLim(1) h.YLim(2)],'r--');
for i=2:4
  h.Children(i).LineWidth = 2;
  h.Children(i).MarkerFaceColor = 'w';
end
h.FontSize=14;

% AIC with correction for finite samples
% n is number of estimated data points
% V is number of free param
% L is -1 * log-likelihood
% 2*L + 2*V + (2*V*(V+1)) / (n-V-1)

% count up the number of recall events, this is the number of data points
% we add n_trials because the model counts recall termination as a
% data point; it tries to predict this just like it tries to
% predict identity of recalled items
ndata = sum(fstruct.data.recalls>0,'all') + n_trials;

% here only B_rec is free
V = 1;
for i=1:length(LL_naive)
  AIC_naive(i) = 2*LL_naive(i) + 2*V + ((2*V*(V+1)) / (ndata-V-1));
end

% B_rec and neural scaling are free
V = 2;
for i=1:length(LL_sweep_neural)
  AIC_sweep_neural(i) = 2*LL_sweep_neural(i) + 2*V + ((2*V*(V+1)) / (ndata-V-1));
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
fprintf('\nComparison of neurally naive and neurally informed models\n');
fprintf('weighted AIC for neurally naive model: %f\n', wAIC(1));
fprintf('weighted AIC for neurally informed model: %f\n', wAIC(2));