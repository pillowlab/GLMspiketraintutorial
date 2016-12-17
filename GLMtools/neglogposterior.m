function [negLP,grad,H] = neglogposterior(prs,negloglifun,Cinv)
% [negLP,grad,H] = neglogposterior(prs,negloglifun,Cinv)
%
% Compute negative log-posterior given a negative log-likelihood function
% and zero-mean Gaussian prior with inverse covariance 'Cinv'.
%
% Inputs:
 %   prs [d x 1] - parameter vector
%    negloglifun - handle for negative log-likelihood function
%   Cinv [d x d] - response (spike count per time bin)
%
% Outputs:
%          negLP - negative log posterior
%   grad [d x 1] - gradient 
%      H [d x d] - Hessian (second deriv matrix)

% Compute negative log-posterior by adding quadratic penalty to log-likelihood

switch nargout

    case 1  % evaluate function
        negLP = negloglifun(prs) + .5*prs'*Cinv*prs;
    
    case 2  % evaluate function and gradient
        [negLP,grad] = negloglifun(prs);
        negLP = negLP + .5*prs'*Cinv*prs;        
        grad = grad + Cinv*prs;

    case 3  % evaluate function and gradient
        [negLP,grad,H] = negloglifun(prs);
        negLP = negLP + .5*prs'*Cinv*prs;        
        grad = grad + Cinv*prs;
        H = H + Cinv;
end

