function [cdist] = cosdist_onetomany(v1,M)
% [cdist] = cosdist_onetomany(v1,M);
%
% cosdist of one vector to another
%
%

% make sure v is a column vectors
[r1,c1] = size(v1);
[r2,c2] = size(M);
if c1 > 1 
  error('this cosine dist function can only handle column vectors');
end

% make sure x1 and x2 are of the same length
if r1 ~= r2
  error('cosine dist input vects must be of same length');
end
  
X = [v1,M];

% if any one vector is close to zero, disp
Xnorm = sqrt(sum(X.^2));
%if min(Xnorm) <= eps(max(Xnorm))
%  disp('vector input to cosine distance is near-zero');
%  cdist = zeros(1,c2);
%else

% now generate cosine dist of v1 to M
vM = v1 * ones(c2,1)';  
cdist = sum(M.*vM) ./ (sqrt(sum(M.*M)) .* sqrt(sum(vM.*vM)));

cdist(isnan(cdist)) = 0;

%end




