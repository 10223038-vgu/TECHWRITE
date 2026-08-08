function params = trainUnfoldedLASSoft(variant, trainData, K1, K2, maxEpochs, learnRate, miniBatchSize, initTemp)
% TRAINUNFOLDEDLASSOFT Trains the Option A (soft-relaxation) Unfolded-A
% or Unfolded-B variant end-to-end against true bits. Same training
% objective/structure as trainUnfoldedLAS.m (Option B) -- only the
% learnable parameters differ: unconstrained log-temperatures instead of
% step sizes, feeding unfoldedLASCoreSoft.m's differentiable softmax
% relaxation instead of Option B's detached hard decision.
%
% initTemp - initial temperature (same for every layer), converted to
% the unconstrained raw parameter via rawTemp = log(initTemp). Smaller
% initTemp starts closer to a hard decision (sharper softmax, closer to
% classical behavior, but weaker gradient signal for the optimizer to
% work with); larger initTemp starts softer (better gradient flow
% initially, but the search itself deviates further from classical
% behavior before training has had a chance to sharpen it back down).
% Default 0.5 is a middle-ground starting point, not a tuned value --
% treat as a hyperparameter to revisit if training struggles.

if nargin < 5 || isempty(maxEpochs),      maxEpochs = 20;   end
if nargin < 6 || isempty(learnRate),      learnRate = 1e-2; end
if nargin < 7 || isempty(miniBatchSize),  miniBatchSize = 32; end
if nargin < 8 || isempty(initTemp),       initTemp = 0.5;   end

Nt = size(trainData(1).H, 2);
M = trainData(1).M;
B = log2(M);

rawTempInit = log(initTemp);
rawTemps1 = dlarray(rawTempInit * ones(K1, 1));
avgG1 = []; avgSqG1 = [];

if strcmpi(variant, 'B')
    rawTemps2 = dlarray(rawTempInit * ones(K2, 1));
    avgG2 = []; avgSqG2 = [];
elseif strcmpi(variant, 'A')
    readoutInSize = 4*Nt;
    readoutOutSize = Nt*B;
    W_readout = dlarray(0.01*randn(readoutOutSize, readoutInSize));
    b_readout = dlarray(zeros(readoutOutSize, 1));
    avgGW = []; avgSqGW = [];
    avgGb = []; avgSqGb = [];
else
    error('trainUnfoldedLASSoft:badVariant', 'variant must be ''A'' or ''B''.');
end

nSamples = numel(trainData);
iteration = 0;
totalBatches = ceil(nSamples/miniBatchSize) * maxEpochs;
ticStart = tic;

for epoch = 1:maxEpochs
    order = randperm(nSamples);
    epochLoss = 0; nBatches = 0;

    for startIdx = 1:miniBatchSize:nSamples
        batchIdx = order(startIdx:min(startIdx+miniBatchSize-1, nSamples));
        iteration = iteration + 1;

        if strcmpi(variant, 'B')
            [loss, grad1, grad2] = dlfeval(@lossAndGradBSoft, trainData(batchIdx), rawTemps1, rawTemps2);
            [rawTemps1, avgG1, avgSqG1] = adamupdate(rawTemps1, grad1, avgG1, avgSqG1, iteration, learnRate);
            [rawTemps2, avgG2, avgSqG2] = adamupdate(rawTemps2, grad2, avgG2, avgSqG2, iteration, learnRate);
        else
            [loss, grad1, gradW, gradb] = dlfeval(@lossAndGradASoft, trainData(batchIdx), rawTemps1, W_readout, b_readout);
            [rawTemps1, avgG1, avgSqG1] = adamupdate(rawTemps1, grad1, avgG1, avgSqG1, iteration, learnRate);
            [W_readout, avgGW, avgSqGW] = adamupdate(W_readout, gradW, avgGW, avgSqGW, iteration, learnRate);
            [b_readout, avgGb, avgSqGb] = adamupdate(b_readout, gradb, avgGb, avgSqGb, iteration, learnRate);
        end

        epochLoss = epochLoss + double(gather(extractdata(loss)));
        nBatches = nBatches + 1;

        % Progress printing: previously this loop printed NOTHING until an
        % entire epoch finished, so a slow (but working) run was visually
        % indistinguishable from a hung one. Print every 10 mini-batches
        % with elapsed time and an ETA, computed from the actual observed
        % per-iteration rate (Option A's fully-differentiable graph is
        % much more expensive per sample than Option B's -- see
        % THEORY_UNFOLDING.md -- so a generic time estimate isn't useful;
        % this uses YOUR machine's actual measured speed instead).
        if mod(iteration, 10) == 0 || iteration == 1
            elapsed = toc(ticStart);
            rate = elapsed / iteration;   % seconds per mini-batch, measured
            eta = rate * (totalBatches - iteration);
            fprintf('  [iter %d/%d] elapsed=%.0fs, ~%.2fs/batch, ETA=%.0fs (%.1f min)\n', ...
                iteration, totalBatches, elapsed, rate, eta, eta/60);
        end
    end

    if strcmpi(variant, 'B')
        curTemps1 = exp(extractdata(rawTemps1))';
        curTemps2 = exp(extractdata(rawTemps2))';
        fprintf('[Unfolded-%s-Soft] Epoch %d/%d: mean BCE loss = %.4f | T1=[%s] T2=[%s]\n', ...
            variant, epoch, maxEpochs, epochLoss/nBatches, num2str(curTemps1, '%.3f '), num2str(curTemps2, '%.3f '));
    else
        curTemps1 = exp(extractdata(rawTemps1))';
        fprintf('[Unfolded-%s-Soft] Epoch %d/%d: mean BCE loss = %.4f | T1=[%s]\n', ...
            variant, epoch, maxEpochs, epochLoss/nBatches, num2str(curTemps1, '%.3f '));
    end
end

if strcmpi(variant, 'B')
    params.temps1 = exp(extractdata(rawTemps1));
    params.temps2 = exp(extractdata(rawTemps2));
else
    params.temps1 = exp(extractdata(rawTemps1));
    params.W_readout = extractdata(W_readout);
    params.b_readout = extractdata(b_readout);
end
end

function [loss, grad1, grad2] = lossAndGradBSoft(batch, rawTemps1, rawTemps2)
loss = dlarray(0);
for i = 1:numel(batch)
    s = batch(i);
    LLR = unfoldedSoftLAS_B_Soft(s.y, s.H, s.sigma2, s.M, s.xr0, rawTemps1, rawTemps2);
    loss = loss + bceFromLLRSoft(LLR, s.trueBits);
end
loss = loss / numel(batch);
[grad1, grad2] = dlgradient(loss, rawTemps1, rawTemps2);
end

function [loss, grad1, gradW, gradb] = lossAndGradASoft(batch, rawTemps1, W_readout, b_readout)
loss = dlarray(0);
for i = 1:numel(batch)
    s = batch(i);
    LLR = unfoldedSoftLAS_A_Soft(s.y, s.H, s.sigma2, s.M, s.xr0, rawTemps1, W_readout, b_readout);
    loss = loss + bceFromLLRSoft(LLR, s.trueBits);
end
loss = loss / numel(batch);
[grad1, gradW, gradb] = dlgradient(loss, rawTemps1, W_readout, b_readout);
end

function loss = bceFromLLRSoft(LLR, trueBits)
logit = -LLR;
target = dlarray(trueBits);
loss = mean(max(logit,0) - logit.*target + log(1+exp(-abs(logit))), 'all');
end
