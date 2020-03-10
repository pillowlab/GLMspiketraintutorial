% tutorial4_regularization_PoissonGLM.m
%
% This is an interactive tutorial covering regularization for Poisson GLMs,
% namely maximum a priori ('MAP') estimation of the linear filter
% parameters under a Gaussian prior.
%
% We'll consider two simple regularization methods:
%
% 1. Ridge regression - corresponds to maximum a posteriori (MAP)
%                       estimation under an iid Gaussian prior on the
%                       filter coefficients. 
%
% 2. L2 smoothing prior - using to an iid Gaussian prior on the
%                         pairwise-differences of the filter(s).
%
% Data: from Uzzell & Chichilnisky 2004; see README file for details. 
%
% Last updated: Mar 10, 2020 (JW Pillow)

% Tutorial instructions: Execute each section below separately using
% cmd-enter. For detailed suggestions on how to interact with this
% tutorial, see header material in tutorial1_PoissonGLM.m

%% ====  1. Load the raw data ============

% ------------------------------------------------------------------------
% Be sure to unzip the data file data_RGCs.zip
% (http://pillowlab.princeton.edu/data/data_RGCs.zip) and place it in 
% this directory before running the tutorial.  
% ------------------------------------------------------------------------
% (Data from Uzzell & Chichilnisky 2004):
datdir = 'data_RGCs/';  % directory where stimulus lives
load([datdir, 'Stim']);    % stimulus (temporal binary white noise)
load([datdir,'stimtimes']); % stim frame times in seconds (if desired)
load([datdir, 'SpTimes']); % load spike times (in units of stim frames)
ncells = length(SpTimes);  % number of neurons (4 for this dataset).
% Neurons #1-2 are OFF, #3-4 are ON.
% -------------------------------------------------------------------------

addpath GLMtools; % add directory with log-likelihood functions

% Pick a cell to work with
cellnum = 3; % (1-2 are OFF cells; 3-4 are ON cells).
tsp = SpTimes{cellnum};

% Compute some basic statistics on the stimulus
dtStim = (stimtimes(2)-stimtimes(1)); % time bin size for stimulus (s)

% See tutorial 1 for some code to visualize the raw data!

%% == 2. Upsample to get finer timescale representation of stim and spikes === 

% The need to regularize GLM parameter estimates is acute when we don't
% have enough data relative to the number of parameters we're trying to
% estimate, or when using correlated (eg naturalistic) stimuli, since the
% stimuli don't have enough power at all frequencies to estimate all
% frequency components of the filter. To simulate that setting we will
% consider the binary white-noise stimulus sampled on a finer time lattice
% than the original stimulus (resulting in 1.6 ms time bins instead of 8ms
% bins defined by the stimulus refresh rate).
%
% For speed of our code and to illustrate the advantages of regularization,
% let's use only a reduced (5-minute) portion of the dataset:
nT = 120*60*1;  % # of time bins for 1 minute of data 
Stim = Stim(1:nT); % reduce stimulus to selected time bins
tsp = tsp(tsp<nT*dtStim); % reduce spikes 

% Now upsample to finer temporal grid
upsampfactor = 5; % divide each time bin by this factor
dtStimhi = dtStim/upsampfactor; % use bins 100 time bins finer
ttgridhi = (dtStimhi/2:dtStimhi:nT*dtStim)'; % fine time grid for upsampled stim
Stimhi = interp1((1:nT)*dtStim,Stim,ttgridhi,'nearest','extrap');
nThi = nT*upsampfactor;  % length of upsampled stimulus

% Visualize the upsampled data.
subplot(211);
iiplot = 1:(60*upsampfactor); % bins of stimulus to plot
ttplot = iiplot*dtStimhi; % time bins of stimulus
plot(ttplot,Stimhi(iiplot), 'linewidth', 2);  axis tight;
title('raw stimulus (fine time bins)');
ylabel('stim intensity');
% Should notice stimulus now constant for many bins in a row

% Bin the spike train and replot binned counts
sps = hist(tsp,ttgridhi)';
subplot(212);
stem(ttplot,sps(iiplot));
title('binned spike counts');
ylabel('spike count'); xlabel('time (s)');
set(gca,'xlim', ttplot([1 end]));
% Now maximum 1 spike per bin!

%%  3. Divide data into "training" and "test" sets for cross-validation

trainfrac = .8;  % fraction of data to use for training
ntrain = ceil(nThi*trainfrac);  % number of training samples
ntest = nThi-ntrain; % number of test samples
iitest = 1:ntest; % time indices for test
iitrain = ntest+1:nThi;   % time indices for training
stimtrain = Stimhi(iitrain,:); % training stimulus
stimtest = Stimhi(iitest,:); % test stimulus
spstrain = sps(iitrain,:);
spstest =  sps(iitest,:);

fprintf('Dividing data into training and test sets:\n');
fprintf('Training: %d samples (%d spikes) \n', ntrain, sum(spstrain));
fprintf('    Test: %d samples (%d spikes)\n', ntest, sum(spstest));

% Set the number of time bins of stimulus to use for predicting spikes
ntfilt = 20*upsampfactor;  

% build the design matrix, training data
Xtrain = [ones(ntrain,1), ... % constant column of ones
    hankel([zeros(ntfilt-1,1);stimtrain(1:end-ntfilt+1)], ... 
    stimtrain(end-ntfilt+1:end))]; % stimulus

% Build design matrix for test data
Xtest = [ones(ntest,1), hankel([zeros(ntfilt-1,1); stimtest(1:end-ntfilt+1)], ...
    stimtest(end-ntfilt+1:end))];

%% === 4. Fit poisson GLM using ML ====================

% Compute maximum likelihood estimate (using 'fminunc' instead of 'glmfit')
sta = (Xtrain'*spstrain)/sum(spstrain); % compute STA for initialization

% -- Set options --- 
% opts = optimset('Gradobj','on','Hessian','on','display','iter');  % OLD VERION
opts = optimoptions('fminunc','algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective','display','iter');

% -- Make loss function and minimize -----
lossfun = @(prs)neglogli_poissGLM(prs,Xtrain,spstrain,dtStimhi); % set negative log-likelihood as loss func
filtML = fminunc(lossfun,sta,opts);

ttk = (-ntfilt+1:0)*dtStimhi;
h = plot(ttk,ttk*0,'k', ttk,filtML(2:end)); 
set(h(2), 'linewidth',2); axis tight;
xlabel('time before spike'); ylabel('coefficient');
title('Maximum likelihood filter estimate'); 

% Looks bad due to lack of regularization!

%% === 5. Ridge regression prior ======================

% Now let's regularize by adding a penalty on the sum of squared filter
% coefficients w(i) of the form:   
%       penalty(lambda) = lambda*(sum_i w(i).^2),
% where lambda is known as the "ridge" parameter.  As noted in tutorial3,
% this is equivalent to placing an iid zero-mean Gaussian prior on the RF
% coefficients with variance equal to 1/lambda. Lambda is thus the inverse
% variance or "precision" of the prior.

% To set lambda, we'll try a grid of values and use
% cross-validation (test error) to select which is best.  

% Set up grid of lambda values (ridge parameters)
lamvals = 2.^(0:10); % it's common to use a log-spaced set of values
nlam = length(lamvals);

% Precompute some quantities (X'X and X'*y) for training and test data
Imat = eye(ntfilt+1); % identity matrix of size of filter + const
Imat(1,1) = 0; % remove penalty on constant dc offset

% Allocate space for train and test errors
negLtrain = zeros(nlam,1);  % training error
negLtest = zeros(nlam,1);   % test error
w_ridge = zeros(ntfilt+1,nlam); % filters for each lambda

% Define train and test log-likelihood funcs
negLtrainfun = @(prs)neglogli_poissGLM(prs,Xtrain,spstrain,dtStimhi); 
negLtestfun = @(prs)neglogli_poissGLM(prs,Xtest,spstest,dtStimhi); 

% Now compute MAP estimate for each ridge parameter
wmap = filtML; % initialize parameter estimate
clf; plot(ttk,ttk*0,'k'); hold on; % initialize plot
for jj = 1:nlam
    
    % Compute ridge-penalized MAP estimate
    Cinv = lamvals(jj)*Imat; % set inverse prior covariance
    lossfun = @(prs)neglogposterior(prs,negLtrainfun,Cinv);
    wmap = fminunc(lossfun,wmap,opts);
    
    % Compute negative logli
    negLtrain(jj) = negLtrainfun(wmap); % training loss
    negLtest(jj) = negLtestfun(wmap); % test loss
    
    % store the filter
    w_ridge(:,jj) = wmap;
    
    % plot it
    plot(ttk,wmap(2:end),'linewidth', 2); 
    title(['ridge estimate: lambda = ', num2str(lamvals(jj))]);
    xlabel('time before spike (s)'); drawnow; pause(0.5);
 
end
hold off;
% note that the esimate "shrinks" down as we increase lambda

%% Plot filter estimates and errors for ridge estimates

subplot(222);
plot(ttk,w_ridge(2:end,:)); axis tight;  
title('all ridge estimates');
subplot(221);
semilogx(lamvals,-negLtrain,'o-', 'linewidth', 2);
title('training logli');
subplot(223); 
semilogx(lamvals,-negLtest,'-o', 'linewidth', 2);
xlabel('lambda');
title('test logli');

% Notice that training error gets monotonically worse as we increase lambda
% However, test error has an dip at some optimal, intermediate value.

% Determine which lambda is best by selecting one with lowest test error 
[~,imin] = min(negLtest);
filt_ridge= w_ridge(2:end,imin);
subplot(224);
plot(ttk,ttk*0, 'k--', ttk,filt_ridge,'linewidth', 2);
xlabel('time before spike (s)'); axis tight;
title('best ridge estimate');


%% === 6. L2 smoothing prior ===========================

% Use penalty on the squared differences between filter coefficients,
% penalizing large jumps between successive filter elements. This is
% equivalent to placing an iid zero-mean Gaussian prior on the increments
% between filter coeffs.  (See tutorial 3 for visualization of the prior
% covariance).

% This matrix computes differences between adjacent coeffs
Dx1 = spdiags(ones(ntfilt,1)*[-1 1],0:1,ntfilt-1,ntfilt); 
Dx = Dx1'*Dx1; % computes squared diffs

% Select smoothing penalty by cross-validation 
lamvals = 2.^(1:14); % grid of lambda values (ridge parameters)
nlam = length(lamvals);

% Embed Dx matrix in matrix with one extra row/column for constant coeff
D = blkdiag(0,Dx); 

% Allocate space for train and test errors
negLtrain_sm = zeros(nlam,1);  % training error
negLtest_sm = zeros(nlam,1);   % test error
w_smooth = zeros(ntfilt+1,nlam); % filters for each lambda

% Now compute MAP estimate for each ridge parameter
clf; plot(ttk,ttk*0,'k'); hold on; % initialize plot
wmap = filtML; % initialize with ML fit
for jj = 1:nlam
    
    % Compute MAP estimate
    Cinv = lamvals(jj)*D; % set inverse prior covariance
    lossfun = @(prs)neglogposterior(prs,negLtrainfun,Cinv);
    wmap = fminunc(lossfun,wmap,opts);
    
    % Compute negative logli
    negLtrain_sm(jj) = negLtrainfun(wmap); % training loss
    negLtest_sm(jj) = negLtestfun(wmap); % test loss
    
    % store the filter
    w_smooth(:,jj) = wmap;
    
    % plot it
    plot(ttk,wmap(2:end),'linewidth',2);
    title(['smoothing estimate: lambda = ', num2str(lamvals(jj))]);
    xlabel('time before spike (s)'); drawnow; pause(.5);
 
end
hold off;

%% Plot filter estimates and errors for smoothing estimates

subplot(222);
plot(ttk,w_smooth(2:end,:)); axis tight;  
title('all smoothing estimates');
subplot(221);
semilogx(lamvals,-negLtrain_sm,'o-', 'linewidth', 2);
title('training LL');
subplot(223); 
semilogx(lamvals,-negLtest_sm,'-o', 'linewidth', 2);
xlabel('lambda');
title('test LL');

% Notice that training error gets monotonically worse as 5we increase lambda
% However, test error has an dip at some optimal, intermediate value.

% Determine which lambda is best by selecting one with lowest test error 
[~,imin] = min(negLtest_sm);
filt_smooth= w_smooth(2:end,imin);
subplot(224);
h = plot(ttk,ttk*0, 'k--', ttk,filt_ridge,...
    ttk,filt_smooth,'linewidth', 1);
xlabel('time before spike (s)'); axis tight;
title('best smoothing estimate');
legend(h(2:3), 'ridge', 'L2 smoothing', 'location', 'northwest');
% clearly the "L2 smoothing" filter looks better by eye!

% Last, lets see which one actually achieved lower test error
fprintf('\nBest ridge test LL:      %.5f\n', -min(negLtest));
fprintf('Best smoothing test LL:  %.5f\n', -min(negLtest_sm));


%% Advanced exercise:
% --------------------
%
% 1. Repeat of the above, but incorporate spike history filters as in
% tutorial2. Use a different smoothing hyperparamter for the spike-history
% / coupling filters than for the stim filter. In this case one needs to
% build a block diagonal prior covariance, with one block for each group of
% coefficients.
 