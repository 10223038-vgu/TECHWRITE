function [seqNet, trainInfo] = trainSeqModel(datasetFile, mlpNet, layerType, numHiddenUnits, maxEpochs)
% TRAINSEQMODEL Generalized version of trainGRU.m that swaps the
% recurrent layer type, used to build the Fig. 12 baselines (LSTM,
% Bi-LSTM, GRU) on top of the same MLP rough-LLR estimate, exactly as
% described in Section V-C: "we consider an LSTM/Bi-LSTM/GRU layer
% with 800 hidden units, a dropout layer with a dropout probability
% of 0.2 with ReLu activation function, and is trained for 500
% epochs."
%
%   layerType: 'gru' | 'lstm' | 'bilstm'
%
% trainGRU.m is kept as a thin backward-compatible wrapper around
% this function with layerType = 'gru' (but note trainGRU.m uses the
% paper's Deep-LAS-specific hyperparameters -- 2 layers, V=100 units,
% dropout 0.01, 40 epochs -- NOT the Fig. 12 benchmark settings; call
% trainSeqModel directly with 'gru',800,500 if you want the Fig. 12
% GRU baseline instead of the Deep LAS GRU block).

if nargin < 4 || isempty(numHiddenUnits), numHiddenUnits = 800; end
if nargin < 5 || isempty(maxEpochs),      maxEpochs = 500;      end

data = load(datasetFile);
X = data.Xm_all;
Xn = X ./ max(abs(X), [], 1);
Y = data.LLR_all;
N = size(X, 2);

llrRough = mlpNet(Xn);
llrRoughSummary = mean(llrRough, 1);
UG = [Xn; llrRoughSummary];

XTrainSeq = cell(N,1);
for i = 1:N
    XTrainSeq{i} = UG(:, i);
end
YTrainAll = Y.';

nVal = round(0.2*N);
valIdx = randperm(N, nVal);
trainIdx = setdiff(1:N, valIdx);

switch lower(layerType)
    case 'gru'
        recurrentLayer = gruLayer(numHiddenUnits, 'OutputMode', 'last');
    case 'lstm'
        recurrentLayer = lstmLayer(numHiddenUnits, 'OutputMode', 'last');
    case 'bilstm'
        recurrentLayer = bilstmLayer(numHiddenUnits, 'OutputMode', 'last');
    otherwise
        error('trainSeqModel:badType', 'layerType must be gru, lstm, or bilstm.');
end

layers = [ ...
    sequenceInputLayer(size(UG,1))
    recurrentLayer
    dropoutLayer(0.2)
    reluLayer
    fullyConnectedLayer(size(Y,1))
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', 40, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XTrainSeq(valIdx), YTrainAll(valIdx, :)}, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'none');

[seqNet, trainInfo] = trainNetwork(XTrainSeq(trainIdx), YTrainAll(trainIdx, :), layers, options);

fprintf('%s baseline trained: %d hidden units, %d epochs.\n', ...
    upper(layerType), numHiddenUnits, maxEpochs);
end
