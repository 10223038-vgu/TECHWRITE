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
    % BUG THIS FIXES: every earlier version of this stack set
    % OutputMode='last' on EVERY GRU layer, including non-final ones.
    % 'last' collapses the full Lseq-timestep sequence down to a single
    % vector -- correct for the FINAL layer (needed before the regression
    % head), but if an earlier layer also collapses, everything after it
    % (including any subsequent "GRU" layer) only ever sees a length-1
    % sequence and can no longer use the correlation structure across
    % timesteps at all. With numGRULayers=2 that meant layer 1 did all
    % the real sequence processing and layer 2 was a wasted no-op that
    % could only degrade the signal (extra dropout/relu/randomly-init'd
    % single-step recurrence) without adding any real depth -- exactly
    % the kind of bug that would leave Direction 1 badly underperforming
    % even after the data and evaluation-distribution fixes.
    if Lyr < numGRULayers
        outputMode = 'sequence';   % pass the full timestep sequence onward
    else
        outputMode = 'last';        % only the final layer collapses to a vector
    end
    layers = [layers, ...
        gruLayer(numHiddenUnits, 'OutputMode', outputMode), ...
        dropoutLayer(0.01), ...
        reluLayer]; %#ok<AGROW>
end
layers = [layers, fullyConnectedLayer(outputSize), regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', 40, ...
    'Shuffle', 'every-epoch', ...
    'GradientThreshold', 5, ...
    'ValidationData', {XTrainSeq(valIdx), YTrainAll(valIdx, :)}, ...
    'ValidationFrequency', 30, ...
    'ExecutionEnvironment', 'cpu', ...
    'Verbose', true, ...
    'Plots', 'none');
% NOTE: ExecutionEnvironment forced to 'cpu' to avoid CUDA_ERROR_UNKNOWN
% crashes seen in the original package; switch to 'auto' once your GPU
% driver/toolbox versions are confirmed compatible.
% NOTE: GradientThreshold added because these sequences are Lseq (~64)
% timesteps long, vs. the original codebase's single-timestep GRU
% training -- much more prone to exploding gradients through
% backpropagation-through-time; the original had no need for this.

[gruNet, trainInfo] = trainNetwork(XTrainSeq(trainIdx), YTrainAll(trainIdx, :), layers, options);
end
