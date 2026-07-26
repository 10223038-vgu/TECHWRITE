function [poolNet, trainInfo] = trainPoolingNet(poolData, numHiddenUnits, maxEpochs)
% TRAINPOOLINGNET Train a plain feedforward (permutation-invariant-input)
% regression network on the pooled dataset from buildPooledContextFeatures.m.
%
% Mirrors trainGRUFromSequences.m's overall scale (2 hidden layers,
% dropout, relu, Adam, same MiniBatchSize/validation split) so any
% performance difference vs. the GRU is attributable to the
% architecture/representation, not to a mismatched training budget.
% Being a plain feedforward net (no BPTT over 64 timesteps), this trains
% much faster and more stably than the GRU for the same data/epoch count.

if nargin < 2 || isempty(numHiddenUnits), numHiddenUnits = 128; end
if nargin < 3 || isempty(maxEpochs),      maxEpochs = 60;       end

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
    dropoutLayer(0.05)
    fullyConnectedLayer(numHiddenUnits)
    reluLayer
    dropoutLayer(0.05)
    fullyConnectedLayer(outputSize)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XTrainAll(valIdx, :), YTrainAll(valIdx, :)}, ...
    'ValidationFrequency', 30, ...
    'ExecutionEnvironment', 'cpu', ...
    'Verbose', true, ...
    'Plots', 'none');

[poolNet, trainInfo] = trainNetwork(XTrainAll(trainIdx, :), YTrainAll(trainIdx, :), layers, options);
end
