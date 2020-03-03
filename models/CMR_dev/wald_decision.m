function out_obj = wald_decision(in_obj, param)
% WALD_DECISION
% in_obj
%
% fields required on the param structure:
%
% in_obj.thisresp  - the index of the response that was made 
% in_obj.thisrt    - how long the response took in msec or sec
%
% param.simulation_mode 'predictive' or 'generative'
%   if predictive, in_obj needs to have fields as shown above
%   if generative, in_obj isn't used, can be an empty vector ([])
% param.n_racers - how many decision alternatives (e.g., 2)
% param.range_rt - what is the possible range of response times
% (e.g., [0 3000])
% param.alpha    - threshold, larger values make it harder to respond
% param.sigma    - noise on the racer
% param.drift    - strength / support for that racer, how fast does it race?



% drift is a vector with n_racers elements 
drift = param.drift;

% conversion of parameters 
muvec = param.alpha ./ drift;
lambdavec = (param.alpha.^2) ./ (param.sigma^2); 

% if generative, pick a first-passage time for each racer
% and figure out who won the race
if strcmp(param.simulation_mode,'generative')

  for i=1:param.n_racers
    % HERE RAND VAL IG
    % originally:
    % fpvec(i) = random(pd{i}, 1);
    
    this_mu = muvec(i);
    this_lambda = lambdavec(i);
    
    yr = randn^2;
    xr = this_mu + ((this_mu^2 * yr) / (2 * this_lambda)) - ...
         (this_mu / (2 * this_lambda)) * ...
         sqrt((4*this_mu*this_lambda*yr) + (this_mu^2 * yr^2));
    zr = rand;
    if zr <= (this_mu / (this_mu + xr))
      ig_rand = xr;
    else
      ig_rand = (this_mu^2 / xr);
    end
    
    fpvec(i) = ig_rand;
    if fpvec(i) <= 0
      fpvec(i) = NaN;
    end
  end
  % figure out which one passed first
  [rt, resp] = min(fpvec);
  
  % if rt is greater than rt range
  if rt > param.range_rt(2)
    resp = -1;
    rt = param.range_rt(2);
  end
  
  out_obj.resp = resp;
  out_obj.rt = rt;
  %keyboard
  
else % predictive
  
  allowable_resp = [1:param.n_racers];
  
  % response index is thisresp
  respind = in_obj.thisresp;
  if respind > 0
    thisrt = in_obj.thisrt;
  elseif respind == -1
    thisrt = param.range_rt(2);
  end
  
  % remove actual resp from the list of other responses
  otherrespind = allowable_resp(allowable_resp~=respind);
  % respind = [];
  
  % calc all the CDFs; mcd = one minus cdf 
  %
  % mcd interpretation: Given thisrt, calculate for each racer the
  % probability that it did not finish by this RT.
  for i=1:param.n_racers
    % HERE CDF IG
    % originally:
    % log_mcd(i) = log(1 - cdf(pd{i}, thisrt));    
    this_mu = muvec(i);
    this_lambda = lambdavec(i);
    term1 = sqrt(this_lambda/thisrt) * ((thisrt/this_mu)-1);
    term2 = -1 * sqrt(this_lambda/thisrt) * ((thisrt/this_mu) + 1);
    % ig_cd = normcdf(term1, 0, 1) + ...
    %         exp((2*this_lambda)/this_mu) * ...
    %         normcdf(term2, 0, 1);

    % using phi.c from M. Koshelev 
    ig_cd = phi(term1) + ...
            exp((2*this_lambda)/this_mu) * ...
            phi(term2);

    % consider this, check this
    if ig_cd > 1-eps
      ig_cd = 1-eps;
    end
    
    log_mcd(i) = log(1 - ig_cd);
    
  end
  
  % initialize first-passage vector
  log_fpvec = zeros(1,param.n_racers);
  
  % Given thisrt, calcluate the probability that the winning racer
  % finished at thisrt.
  
  % only do this if there was a winner:
  if respind > 0
    % this uses whichever distribution is appropriate, levy or wald
    % HERE PDF IG
    % originally:
    % log_fpvec(respind) = log(pdf(pd{respind},thisrt));
    this_mu = muvec(respind);
    this_lambda = lambdavec(respind);
    ig_pd = sqrt(this_lambda / (2 * pi * thisrt^3)) ...
        * exp((-this_lambda * (thisrt-this_mu)^2) / (2 * this_mu^2 * thisrt));
    % consider this, check this
    if ig_pd == 0
      ig_pd = eps;
    end
    log_fpvec(respind) = log(ig_pd);
  end

  % The final calculation: The probability that the winner won at
  % thisrt, multiplied by the probability that the other racers
  % hadn't finished yet
  if respind > 0
    out_obj.logl = log_fpvec(respind) + sum(log_mcd(otherrespind));
  end
  
  % if a response wasn't made by the specified deadline, the
  % likelihood of this event is the probability that none of the
  % racers finished by that deadline.
  if respind == -1
    out_obj.logl = sum(log_mcd);
  end
  % if in_obj.thisresp == -1
  %   keyboard
  % end
  % if imag(out_obj.logl)~=0
  %   keyboard;
  % end
  
end % predictive

