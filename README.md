# GLMspiketraintutorial
Simple tutorial on Gaussian and Poisson GLMs for single and multi-neuron spike train data.

This tutorial were prepared for the Society for Neuroscience 2016
"Short Course" on
[Data Science and Data Skills for Neuroscientists](http://www.stat.ucla.edu/~akfletcher/WebSfN.htm),
held in San Diego in Nov, 2016, presented by
[Jonathan Pillow](http://pillowlab.princeton.edu).  The slides used
during the 1-hour short course presentation are available the "slides"
directory.

The tutorial is broken into three pieces which aim to introduce
methods for fitting and simulating Gaussian and Poisson regression
models for neural data. Each is an interactive, self-contained script 
with 'blocks' of code that demonstrate each step in the fitting /
analysis / model comparison / simulation pipeline: 

* **tutorial1_PoissonGLM.m** - illustrates the fitting of a
linear-Gaussian GLM (also known as the 'linear least-squares
regression model') and a Poisson GLM (aka 'linear-nonlinear-Poisson'
model) to single retinal ganglion cell responses to a temporal white
noise stimulus.

* **tutorial2_spikehistcoupledGLM.m** - fitting of an autoregressive
Poisson GLM (i.e., a GLM with spike-history) and a multivariate
autoregressive Poisson GLM (a GLM with spike-history AND coupling
between neurons).

* **tutorial3_regularization.m** - regularizing estimates of GLM
  parameters (to prevent overfitting) via: (1) ridge regression (aka
  "L2 penalty"); and (2) L2 smoothing prior ("graph Laplacian").

Note that the data used for this tutorial is (unfortunately) not yet
publicly available. If you would like access to the dataset needed to
run the three tutorial scripts, please write to pillow at princeton
dot edu.
