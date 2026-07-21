function [gruNet, trainInfo] = trainGRU(datasetFile, mlpNet, numGRULayers, numHiddenUnits, maxEpochs)
% TRAINGRU Train the GRU block of the proposed Deep LAS
% (Section IV-A, Fig. 2, Eqs. 22-23).
%
% The MLP block's rough LLR estimate is concatenated with the
% pre-processed input (U_G = [y'; LLR_MLP], Eq. under Fig.2) to form
% an extended sequence of length 2*Nt+1, each of length N (here we
% treat each of the 2Nt+1 rows as one timestep of a sequence, i.e.
% a "sequence-to-sequence" regression of length 2Nt+1 per sample --
% matches "2Nt+1 sequences each of 1xN length" in the text).
%
%   numGRULayers   - sweep 1,2,3 for Fig. 4b (paper uses 2 sub-blocks)
%   numHiddenUnits - V in the paper (V = 100)
%   maxEpochs      - paper: 40
%
% Requires Deep Learning Toolbox (trainNetwork, gruLayer).

if nargin < 3 || isempty(numGRULayers),   numGRULayers = 2;   end
if nargin < 4 || isempty(numHiddenUnits), numHiddenUnits = 100; end
if nargin < 5 || isempty(maxEpochs),      maxEpochs = 40;      end

data = load(datasetFile);
X = data.Xm_all;                    % (2Nt) x N
Xn = X ./ max(abs(X), [], 1);
Y = data.LLR_all;                   % (Nt*log2M) x N
N = size(X, 2);

% MLP rough estimate (must be same dimension as target LLR per-sample;
% MLP net trained above regresses the full Nt*log2M LLR vector, so we
% reduce/broadcast it to a single "rough LLR summary" row per Eq. 21's
% single-output description, then feed it as one extra channel).
llrRough = mlpNet(Xn);              % (Nt*log2M) x N
llrRoughSummary = mean(llrRough, 1); % 1 x N   (single-neuron MLP output, Eq. 21)

% Build U_G = [y'; LLR_MLP] per sample -> each sample is a
% (2Nt+1) x 1 feature vector; treat each sample as a sequence with
% one timestep of dimension (2Nt+1) for MATLAB's sequence layers
% (adjust here if you want multi-timestep OFDM-block sequences).
UG = [Xn; llrRoughSummary];         % (2Nt+1) x N

XTrainSeq = cell(N,1);
for i = 1:N
    XTrainSeq{i} = UG(:, i);        % (2Nt+1) x 1  (one timestep)
end
YTrainAll = Y.';                    % N x (Nt*log2M) numeric matrix
                                     % (trainNetwork requires a plain
                                     % matrix, NOT a cell array, for
                                     % sequence-to-one regression with
                                     % 'OutputMode','last')

nVal = round(0.2*N);
valIdx = randperm(N, nVal);
trainIdx = setdiff(1:N, valIdx);

layers = sequenceInputLayer(size(UG,1));
for L = 1:numGRULayers
    layers = [layers, ...
        gruLayer(numHiddenUnits, 'OutputMode', 'last'), ...
        dropoutLayer(0.01), ...
        reluLayer]; %#ok<AGROW>
end
layers = [layers, fullyConnectedLayer(size(Y,1)), regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', 40, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XTrainSeq(valIdx), YTrainAll(valIdx, :)}, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'none');

[gruNet, trainInfo] = trainNetwork(XTrainSeq(trainIdx), YTrainAll(trainIdx, :), layers, options);

fprintf('GRU training done: %d layers x %d units.\n', numGRULayers, numHiddenUnits);
end
