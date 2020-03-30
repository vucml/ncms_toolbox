
% set the seed for rng
% uncomment if you want it to always turn out the same way
rng(123)


% Section 1.  Setting model parameters and task details

% set parameters for a basic version of CMR
param = struct();
% context integration rate during encoding
param.B_enc = 0.7;
% context integration rate during recall
param.B_rec = 0.4;
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
n_trials = 300;
max_recalls = LL;
% creating a data structure for our synthetic (model generated) data
data = struct();
data.subject = ones(n_trials,1);
data.recalls = zeros(n_trials,max_recalls);
data.pres_itemnos = repmat([1:LL],n_trials,1);

% Section 2.  Generating synthetic data and plotting summary
% statistics from the data

% low B_rec
data_low = data;
param_low = param;

% use the CMR model to generate 120 trials of synthetic data
seq = generate_cmr(param_low, data_low);
% store the recall sequences in the data structure
data_low.recalls = seq;

% high B_rec
param_high = param;
param_high.B_rec = 0.8;
data_high = data;

% use the CMR model to generate 120 trials of synthetic data
seq = generate_cmr(param_high, data_high);
% store the recall sequences in the data structure
data_high.recalls = seq;


%figpath = '';
figpath = '~/Science/GitHub/chapter-predictive-cmr/figs/';

figure(1); clf;

lc_low = crp(data_low.recalls,data_low.subject,LL);
lc_high = crp(data_high.recalls,data_high.subject,LL);

plot_crp(lc_low);
hold on; 

h = gca;
plot([-5:5],lc_high(:,LL-5:LL+5),'k^-');

h.Children(1).MarkerSize = 10;
h.Children(1).MarkerFaceColor = 'k';
h.Children(1).LineWidth = 1.5;

h.Children(2).MarkerSize = 10;
h.Children(2).MarkerFaceColor = 'w';
h.Children(2).LineWidth = 1.5;

if ~isempty(figpath)
  print('-depsc', fullfile(figpath,'var_crp.eps'));
end
