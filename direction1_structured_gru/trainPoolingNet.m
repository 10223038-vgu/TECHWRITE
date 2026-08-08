function [poolNet, trainInfo] = trainPoolingNet(poolData, numHiddenUnits, maxEpochs, dropoutRate, l2Reg)
% TRAINPOOLINGNET Train a plain feedforward (permutation-invariant-input)
% regression network on the pooled dataset from buildPooledContextFeatures.m.
%
% Mirrors trainGRUFromSequences.m's overall scale (2 hidden layers,
% dropout, relu, Adam, same MiniBatchSize/validation split) so any
% performance difference vs. the GRU is attributable to the
% architecture/representation, not to a mismatched training budget.
% Being a plain feedforward net (no BPTT over 64 timesteps), this trains
% much faster and more stably than the GRU for the same data/epoch count.
%
% dropoutRate, l2Reg - exposed for testing the hypothesis that this net's
% capacity (2x128 hidden units, far bigger than Original's 10-unit MLP +
% 100-unit GRU combined) combined with weak regularization (previously a
% hardcoded 0.05 dropout, default 1e-4 L2) is why it significantly
% UNDERPERFORMS Original below ~12 dB SNR (see run_stability_analysis.m's
% results): at low SNR the training targets (true LLRs) are themselves
% noisiest, which is exactly where a larger, weakly-regularized network
% is most prone to fitting noise instead of signal. Defaults match the
% previous hardcoded behavior for backward compatibility.

if nargin < 2 || isempty(numHiddenUnits), numHiddenUnits = 128; end
if nargin < 3 || isempty(maxEpochs),      maxEpochs = 60;       end
if nargin < 4 || isempty(dropoutRate),    dropoutRate = 0.05;   end
if nargin < 5 || isempty(l2Reg),          l2Reg = 1e-4;         end

XTrainAll = poolData.X;
YTrainAll = poolData.Y;
Ntrain = size(XTrainAll, 1);

nVal = round(0.2*Ntrain);
valIdx = randperm(Ntrain, nVal);
trainIdx = setdiff(1:Ntrain, valIdx);

inputSize = size(XTrainAll, 2);
outputSize = size(YTrainAll, 2);

layers = [
    featureInputLayer(inputSize, 'Normalization', 'none')
    fullyConnectedLayer(numHiddenUnits)
    reluLayer
    dropoutLayer(dropoutRate)
    fullyConnectedLayer(numHiddenUnits)
    reluLayer
    dropoutLayer(dropoutRate)
    fullyConnectedLayer(outputSize)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'L2Regularization', l2Reg, ...
    'ValidationData', {XTrainAll(valIdx, :), YTrainAll(valIdx, :)}, ...
    'ValidationFrequency', 30, ...
    'ExecutionEnvironment', 'cpu', ...
    'Verbose', true, ...
    'Plots', 'none');

[poolNet, trainInfo] = trainNetwork(XTrainAll(trainIdx, :), YTrainAll(trainIdx, :), layers, options);
end
