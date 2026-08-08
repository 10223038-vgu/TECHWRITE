function params = trainUnfoldedLAS(variant, trainData, K1, K2, maxEpochs, learnRate, miniBatchSize)
% TRAINUNFOLDEDLAS Trains Unfolded-A or Unfolded-B end-to-end against
% TRUE TRANSMITTED BITS (binary cross-entropy), NOT distilled from
% softOutputLAS.m's own output the way the original paper's GRU was
% trained (regression to the classical algorithm's LLR). This is the
% core methodological difference Direction 2 is testing: can learning
% the unfolded step sizes directly from ground truth do better than
% either the fixed classical algorithm or a network trained to imitate it?
%
% Uses a manual dlarray/dlfeval/adamupdate training loop (not
% trainNetwork) because this architecture -- a discrete local search with
% a detached candidate-selection step, run once per real dimension per
% layer -- does not fit any off-the-shelf layer type.
%
% PERFORMANCE NOTE: the forward pass loops over samples one at a time
% (not vectorized across a mini-batch) because the discrete candidate
% search is inherently per-sample (candidate costs depend on that
% sample's own Hr/z). This is a correct, working FIRST version, not a
% fast one -- expect training to be slower than Direction 1's pooling
% nets for the same dataset size. Vectorizing across a batch (e.g. via
% pagemtimes-based batched Q/z updates) is a reasonable follow-up
% optimization if this is too slow in practice, not a correctness fix.
%
% Inputs:
%   variant       - 'A' or 'B' (see unfoldedSoftLAS_A.m / _B.m)
%   trainData     - struct array, one element per training sample, each
%                   with fields .y, .H, .sigma2, .M, .xr0, .trueBits
%                   (Nt x log2(M), same layout as symbolsToBits.m)
%   K1, K2        - number of unfolded layers for Algorithm 1 / Algorithm
%                   2 respectively (K2 unused for variant 'A')
%   maxEpochs, learnRate, miniBatchSize - standard training controls
%
% Output:
%   params - struct with the learned parameters:
%     variant 'A': .alphas1, .W_readout, .b_readout
%     variant 'B': .alphas1, .alphas2

if nargin < 5 || isempty(maxEpochs),      maxEpochs = 20;   end
if nargin < 6 || isempty(learnRate),      learnRate = 1e-2; end
if nargin < 7 || isempty(miniBatchSize),  miniBatchSize = 32; end

Nt = size(trainData(1).H, 2);
M = trainData(1).M;
B = log2(M);

% --- Initialize learnable parameters ---
% alpha=1 for every layer reproduces the classical algorithm's per-layer
% behavior exactly, so this is a principled starting point (not a random
% one) -- training starts AT the classical algorithm and learns whether
% deviating from unit step size helps.
alphas1 = dlarray(ones(K1, 1));
avgG1 = []; avgSqG1 = [];

if strcmpi(variant, 'B')
    alphas2 = dlarray(ones(K2, 1));
    avgG2 = []; avgSqG2 = [];
elseif strcmpi(variant, 'A')
    readoutInSize = 4*Nt;      % [xr_final; z_final], each 2*Nt
    readoutOutSize = Nt*B;
    W_readout = dlarray(0.01*randn(readoutOutSize, readoutInSize));
    b_readout = dlarray(zeros(readoutOutSize, 1));
    avgGW = []; avgSqGW = [];
    avgGb = []; avgSqGb = [];
else
    error('trainUnfoldedLAS:badVariant', 'variant must be ''A'' or ''B''.');
end

nSamples = numel(trainData);
iteration = 0;

for epoch = 1:maxEpochs
    order = randperm(nSamples);
    epochLoss = 0; nBatches = 0;

    for startIdx = 1:miniBatchSize:nSamples
        batchIdx = order(startIdx:min(startIdx+miniBatchSize-1, nSamples));
        iteration = iteration + 1;

        if strcmpi(variant, 'B')
            [loss, grad1, grad2] = dlfeval(@lossAndGradB, trainData(batchIdx), alphas1, alphas2);
            [alphas1, avgG1, avgSqG1] = adamupdate(alphas1, grad1, avgG1, avgSqG1, iteration, learnRate);
            [alphas2, avgG2, avgSqG2] = adamupdate(alphas2, grad2, avgG2, avgSqG2, iteration, learnRate);
        else
            [loss, grad1, gradW, gradb] = dlfeval(@lossAndGradA, trainData(batchIdx), alphas1, W_readout, b_readout);
            [alphas1, avgG1, avgSqG1]   = adamupdate(alphas1, grad1, avgG1, avgSqG1, iteration, learnRate);
            [W_readout, avgGW, avgSqGW] = adamupdate(W_readout, gradW, avgGW, avgSqGW, iteration, learnRate);
            [b_readout, avgGb, avgSqGb] = adamupdate(b_readout, gradb, avgGb, avgSqGb, iteration, learnRate);
        end

        epochLoss = epochLoss + double(gather(extractdata(loss)));
        nBatches = nBatches + 1;
    end

    fprintf('[Unfolded-%s] Epoch %d/%d: mean BCE loss = %.4f\n', variant, epoch, maxEpochs, epochLoss/nBatches);
end

if strcmpi(variant, 'B')
    params.alphas1 = extractdata(alphas1);
    params.alphas2 = extractdata(alphas2);
else
    params.alphas1 = extractdata(alphas1);
    params.W_readout = extractdata(W_readout);
    params.b_readout = extractdata(b_readout);
end
end

function [loss, grad1, grad2] = lossAndGradB(batch, alphas1, alphas2)
loss = dlarray(0);
for i = 1:numel(batch)
    s = batch(i);
    LLR = unfoldedSoftLAS_B(s.y, s.H, s.sigma2, s.M, s.xr0, alphas1, alphas2);
    loss = loss + bceFromLLR(LLR, s.trueBits);
end
loss = loss / numel(batch);
[grad1, grad2] = dlgradient(loss, alphas1, alphas2);
end

function [loss, grad1, gradW, gradb] = lossAndGradA(batch, alphas1, W_readout, b_readout)
loss = dlarray(0);
for i = 1:numel(batch)
    s = batch(i);
    LLR = unfoldedSoftLAS_A(s.y, s.H, s.sigma2, s.M, s.xr0, alphas1, W_readout, b_readout);
    loss = loss + bceFromLLR(LLR, s.trueBits);
end
loss = loss / numel(batch);
[grad1, gradW, gradb] = dlgradient(loss, alphas1, W_readout, b_readout);
end

function loss = bceFromLLR(LLR, trueBits)
% Sign convention matches the rest of the codebase (e.g.
% simulateBERPooling.m: hardBits = double(LLR < 0)) -- bit=1 <=> LLR<0,
% i.e. logit for "P(bit=1)" is -LLR. Numerically stable BCE-with-logits:
%   BCE(logit,target) = max(logit,0) - logit.*target + log(1+exp(-abs(logit)))
logit = -LLR;
target = dlarray(trueBits);
loss = mean(max(logit,0) - logit.*target + log(1+exp(-abs(logit))), 'all');
end
