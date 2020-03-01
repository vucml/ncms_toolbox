
% demonstration of predictive and generative modeling with CMR

% set parameters for a basic version of CMR
param = struct();
% context integration rate during encoding
param.B_enc = 0.6;
% context integration rate during recall
param.B_rec = 0.5;
% primacy learning rate boost
param.P1 = 3;
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

% calculate and plot some basic summary statistics
% uses EMBAM functions

figure(1); clf;
plot_spc(spc(seq,data.subject,LL));
figure(2); clf;
plot_crp(crp(seq,data.subject,LL));

% a simple test of parameter recovery
% create 11 model variants with different values of 
% beta rec, which controls temporal reinstatement
% evaluate likelihood of the synthetic data for each model variant
fstruct.data = data;
B_rec_vals = [0:0.1:1];
for i=1:length(B_rec_vals)
  temp_param = param;
  temp_param.B_rec = B_rec_vals(i);
  err(i) = eval_param_cmr(temp_param, fstruct);
end

% demonstrate best-fitting value matching the generating value
figure(3); clf;
plot(B_rec_vals,err,'ko-');
xlabel('$\beta$_{rec} value')
ylabel('|log-likelihood|')
h = gca;
h.Children(1).LineWidth = 2;
h.Children(1).MarkerFaceColor = 'w';
h.FontSize=14;

% construct variable beta rec model
% beta rec centers on 0.5 with random normal deviations
% truncated at 0 and 1 
base_B_rec = 0.5;
var_B_rec = base_B_rec + randn(n_trials, max_recalls) * 0.1;
var_B_rec(var_B_rec<0) = 0;
var_B_rec(var_B_rec>1) = 1;

% make variable parameter matrix
% beta rec will vary from recall event to recall event
var_param = struct();
var_param(1).name = 'B_rec';
var_param(1).update_level = 'recall_event';
var_param(1).val = var_B_rec;
% this creates a more efficient structure for controlling parameter fluctuations
% trials x events x parameters
var_param = create_param_mat(var_param, n_trials, LL, max_recalls);
% synthetic data with variability in beta rec
seq = generate_cmr(param, data, var_param);
var_data = data;
var_data.recalls = seq;

% turning variable B rec into synthetic neural data
neural_mat = zeros(n_trials, max_recalls);
for i=1:n_trials
  neural_mat(i,:) = (var_B_rec(i,:) - mean(var_B_rec(i,:))) / std(var_B_rec(i,:));
end

fstruct = struct();
fstruct.data = var_data;
% get likelihood for different values of beta rec and neural
% scaling parameter

B_rec_vals = [0:0.1:1];
nsp_vals = [0:0.1:1];
for i=1:length(B_rec_vals)
  for j=1:length(nsp_vals)
    temp_param = param;
    temp_param.B_rec = B_rec_vals(i);
    temp_param.n = nsp_vals(j);
    
    % for a particular candidate value of B_rec
    %fstruct.neural_mat = neural_mat;
    vp_neural = struct();
    vp_neural(1).name = 'B_rec';
    vp_neural(1).update_level = 'recall_event';
    val = temp_param.B_rec + neural_mat * temp_param.n;
    val(val<0) = 0;
    val(val>0) = 1;
    vp_neural(1).val = val;
    
    vp_neural = create_param_mat(vp_neural, n_trials, LL, max_recalls);
    
    fstruct.var_param = vp_neural;
    
    err = eval_param_cmr(temp_param, fstruct);  
    tr_err(i,j) = err;
  end
end

figure(4); clf;
imagesc(tr_err);


