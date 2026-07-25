function [gruNet, trainInfo] = trainGRUFromSequences(seqData, numGRULayers, numHiddenUnits, maxEpochs)
% TRAINGRUFROMSEQUENCES Train a GRU on a pre-built sequence dataset
% (struct with .X cell array, .Y matrix), as produced by
% buildCorrelatedAndShuffledSequences.m or buildIIDSequences.m.
%
% This mirrors the network architecture and training options in
% original/deep-las-matlab/trainGRU.m exactly, just decoupled from that
% file's own (flat, i.i.d.-only) dataset-building logic, so the three
% ablation arms (correlated / shuffled / iid) are trained identically
% except for their input data.

if nargin < 2 || isempty(numGRULayers),   numGRULayers = 2;   end
if nargin < 3 || isempty(numHiddenUnits), numHiddenUnits = 100; end
if nargin < 4 || isempty(maxEpochs),      maxEpochs = 40;      end

XTrainSeq = seqData.X;
YTrainAll = seqData.Y;
Ntrain = numel(XTrainSeq);

nVal = round(0.2*Ntrain);
valIdx = randperm(Ntrain, nVal);
trainIdx = setdiff(1:Ntrain, valIdx);

inputSize = size(XTrainSeq{1}, 1);
outputSize = size(YTrainAll, 2);

layers = sequenceInputLayer(inputSize);
for Lyr = 1:numGRULayers
    layers = [layers, ...
        gruLayer(numHiddenUnits, 'OutputMode', 'last'), ...
        dropoutLayer(0.01), ...
        reluLayer]; %#ok<AGROW>
end
layers = [layers, fullyConnectedLayer(outputSize), regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', 40, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XTrainSeq(valIdx), YTrainAll(valIdx, :)}, ...
    'ValidationFrequency', 30, ...
    'ExecutionEnvironment', 'cpu', ...
    'Verbose', true, ...
    'Plots', 'none');
% NOTE: ExecutionEnvironment forced to 'cpu' to avoid CUDA_ERROR_UNKNOWN
% crashes seen in the original package; switch to 'auto' once your GPU
% driver/toolbox versions are confirmed compatible.

[gruNet, trainInfo] = trainNetwork(XTrainSeq(trainIdx), YTrainAll(trainIdx, :), layers, options);
end
