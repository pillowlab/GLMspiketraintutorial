function [neglogli, dL, H] = neglogli_poissGLM(prs,XX,YY,dtbin)
% [neglogli, dL, H] = Loss_GLM_logli_exp(prs,XX);
%
% Compute negative log-likelihood of data undr Poisson GLM model with
% exponential nonlinearity
%
% Inputs:
%   prs [d x 1] - parameter vector
%    XX [T x d] - design matrix
%    YY [T x 1] - response (spike count per time bin)
% dtbin [1 x 1] - time bin size used 
%
% Outputs:
%   neglogli   = negative log likelihood of spike train
%   dL [d x 1] = gradient 
%   H  [d x d] = Hessian (second deriv matrix)

% Compute GLM filter output and condititional intensity
vv = XX*prs; % filter output
rr = exp(vv)*dtbin; % conditional intensity (per bin)

% ---------  Compute log-likelihood -----------
Trm1 = -vv'*YY; % spike term from Poisson log-likelihood
Trm0 = sum(rr);  % non-spike term 
neglogli = Trm1 + Trm0;

% ---------  Compute Gradient -----------------
if (nargout > 1)
    dL1 = -XX'*YY; % spiking term (the spike-triggered average)
    dL0 = XX'*rr; % non-spiking term
    dL = dL1+dL0;    
end

% ---------  Compute Hessian -------------------
if nargout > 2
    H = XX'*bsxfun(@times,XX,rr); % non-spiking term 
end

