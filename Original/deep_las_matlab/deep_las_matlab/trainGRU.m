function [gruNet, trainInfo] = trainGRU(datasetFile, mlpNet, numGRULayers, numHiddenUnits, maxEpochs, seqLen)
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
%   seqLen         - NEW: number of consecutive dataset samples pooled
%                    into one multi-timestep training sequence (default
%                    1 = original single-timestep behavior). This is
%                    what lets Fig. 11(a)'s "longer FFT length -> more
%                    pooled input per training sequence -> better LLR
%                    approximation" story actually be reproduced: pass
%                    a larger seqLen to simulate a longer FFT length's
%                    effect on GRU training (see run_fig11_sweeps.m).
%                    The target for each pooled sequence is the LLR of
%                    the LAST sample in the group (standard
%                    sequence-to-one convention). This grouping is a
%                    documented interpretive proxy, not a literal
%                    per-OFDM-subcarrier sequence -- see README.md.
%
% Requires Deep Learning Toolbox (trainNetwork, gruLayer).

if nargin < 3 || isempty(numGRULayers),   numGRULayers = 2;   end
if nargin < 4 || isempty(numHiddenUnits), numHiddenUnits = 100; end
if nargin < 5 || isempty(maxEpochs),      maxEpochs = 40;      end
if nargin < 6 || isempty(seqLen),         seqLen = 1;          end

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

UG = [Xn; llrRoughSummary];         % (2Nt+1) x N

if seqLen <= 1
    % Original single-timestep behavior
    XTrainSeq = cell(N,1);
    for i = 1:N
        XTrainSeq{i} = UG(:, i);
    end
    YTrainAll = Y.';
else
    % Pool seqLen consecutive samples into one multi-timestep sequence;
    % target = LLR of the LAST sample in each pooled group.
    nSeq = floor(N / seqLen);
    XTrainSeq = cell(nSeq, 1);
    YTrainAll = zeros(nSeq, size(Y,1));
    for i = 1:nSeq
        rng_i = (i-1)*seqLen + 1 : i*seqLen;
        XTrainSeq{i} = UG(:, rng_i);          % (2Nt+1) x seqLen
        YTrainAll(i, :) = Y(:, rng_i(end)).'; % target = last timestep's LLR
    end
end
Ntrain = numel(XTrainSeq);

nVal = round(0.2*Ntrain);
valIdx = randperm(Ntrain, nVal);
trainIdx = setdiff(1:Ntrain, valIdx);

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
    'ExecutionEnvironment', 'cpu', ...
    'Verbose', true, ...
    'Plots', 'none');
% NOTE: ExecutionEnvironment is forced to 'cpu' to avoid CUDA_ERROR_UNKNOWN
% crashes; switch to 'auto' once your GPU driver/toolbox versions are
% confirmed compatible.

[gruNet, trainInfo] = trainNetwork(XTrainSeq(trainIdx), YTrainAll(trainIdx, :), layers, options);

fprintf('GRU training done: %d layers x %d units, seqLen=%d.\n', ...
    numGRULayers, numHiddenUnits, seqLen);
end
