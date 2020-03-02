function [cdist] = cosdist(x1,x2)
% [cdist] = cosdist(x1,x2);
%
% cosdist of one vector to another
% if there are NaNs, ignore these dimensions
%

% make sure they are both column vectors
[r1,c1] = size(x1);
[r2,c2] = size(x2);
if c1 > 1 || c2 > 1
  error('this cosine dist function can only handle column vectors');
end

% make sure x1 and x2 are of the same length
if length(x1) ~= length(x2)
  error('cosine dist input vects must be of same length');
end

% if there are NaNs, exclude these dims
x1nan = isnan(x1);
x2nan = isnan(x2);
xnan = or(x1nan,x2nan);
x1 = x1(~xnan);
x2 = x2(~xnan);

X = [x1,x2];

% if any one vector is close to zero, error
Xnorm = sqrt(sum(X.^2));
if min(Xnorm) <= eps(max(Xnorm))
  disp('vector input to cosine distance is near-zero');
  cdist = 0;
else
  cdist = (x1'*x2)/(sqrt(x1'*x1)*sqrt(x2'*x2));
end



