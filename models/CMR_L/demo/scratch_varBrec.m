
% requires demo_neurocog to be run first

LL = 24;
% want to make a lag-CRP split by level of B_rec


lo_B_rec_mask = var_B_rec < 0.5;
hi_B_rec_mask = var_B_rec > 0.5;

lo_lc = crp(var_data.recalls, ...
            var_data.subject, ...
            LL, lo_B_rec_mask);

hi_lc = crp(var_data.recalls, ...
            var_data.subject, ...
            LL, hi_B_rec_mask);

figure(1); clf;

plot([-5:5],lo_lc(:,LL-5:LL+5),'ko-')
hold on;
plot([-5:5],hi_lc(:,LL-5:LL+5),'r^-')

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

