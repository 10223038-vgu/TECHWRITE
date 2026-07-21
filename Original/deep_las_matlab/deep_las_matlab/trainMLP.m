function [mlpNet, trainInfo] = trainMLP(datasetFile, numHiddenLayers, neuronsPerLayer, epochs)
% TRAINMLP Train the MLP block of the proposed Deep LAS
% (Section IV-A, Eqs. 20-21, Algorithm 3).
%
%   numHiddenLayers - H-2 in the paper's notation (sweep 1,2,3 for Fig.4a)
%   neuronsPerLayer  - K in the paper (K = 10 in Section V)
%   epochs           - training epochs (paper: 300)
%
% Trains ONE MLP per LLR bit position is NOT what the paper does;
% the MLP jointly regresses the full LLR vector from xhat_mmse
% (Eq. 19-21), so this trains a single multi-output network.
%
% Requires Deep Learning Toolbox (fitnet) / Statistics and Machine
% Learning Toolbox.

if nargin < 2 || isempty(numHiddenLayers), numHiddenLayers = 2; end
if nargin < 3 || isempty(neuronsPerLayer), neuronsPerLayer = 10; end
if nargin < 4 || isempty(epochs),          epochs = 300;        end

data = load(datasetFile);          % Xm_all, LLR_all
X = data.Xm_all;                   % (2Nt) x N
X = X ./ max(abs(X), [], 1);       % per-sample normalization (Algorithm 3)
Y = data.LLR_all;                  % (Nt*log2M) x N

hiddenSizes = neuronsPerLayer * ones(1, numHiddenLayers);
mlpNet = fitnet(hiddenSizes, 'trainlm');   % Levenberg-Marquardt, per paper
mlpNet.trainParam.epochs = epochs;
mlpNet.trainParam.max_fail = max(epochs, 1000);  % effectively disable early
                                                  % stopping so training runs
                                                  % the full fixed epoch
                                                  % budget, per the paper
                                                  % (must be finite in MATLAB)
mlpNet.divideParam.trainRatio = 0.8;       % 80/20 train/val split (Section IV-A)
mlpNet.divideParam.valRatio   = 0.2;
mlpNet.divideParam.testRatio  = 0.0;

[mlpNet, trainInfo] = train(mlpNet, X, Y);

fprintf('MLP training done: %d hidden layers x %d neurons, final perf = %.4f\n', ...
    numHiddenLayers, neuronsPerLayer, trainInfo.best_perf);
end
