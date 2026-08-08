%% run_fig4_trainingLoss.m
% Reproduces Fig. 4: training loss vs epoch for (a) MLP with
% 1/2/3 fully-connected hidden layers, (b) GRU block with 1/2/3
% GRU layers.
%
% NOTES ON MATCHING THE PAPER'S PLOT STYLE:
%  - trainNetwork's trainInfo.TrainingLoss is logged PER MINI-BATCH
%    ITERATION, not per epoch, which is why a raw plot of it looks
%    like noisy static with an x-axis in the thousands instead of
%    ~40. This script converts it to a per-epoch average so the
%    x-axis and shape match Fig. 4(b).
%  - Our LAS-computed LLR targets are on a different absolute scale
%    than whatever the paper's LLRs were scaled to (our demo saw
%    LLR magnitudes up into the hundreds; the paper's Figs. 5-8 show
%    roughly +/-20-40). That makes our raw MSE loss sit around 1e5
%    instead of ~0.8. We do NOT renormalize targets here (that would
%    require threading an inverse-transform through trainGRU.m,
%    deepLASPredict.m, and simulateBER.m, which is real surgery you
%    don't want done silently). Instead this script gives you both a
%    log-scale view (see the whole decay shape) and a zoomed
%    linear-scale view (see the tail, data-driven range rather than
%    a hardcoded [0 0.8] that would just be empty at our scale).
%    If you DO want targets normalized to make the absolute numbers
%    match the paper's y-axis, say so and I'll thread it through
%    properly rather than patching just the plot.

clear; clc;
M = 4;
dataFile = sprintf('train_%dQAM.mat', M);
if ~isfile(dataFile)
    generateTrainingData(M, 0:2:14, 1000, dataFile);
end

%% (a) MLP: sweep number of FC hidden layers
mlpPerf = cell(1,3);
for nLayers = 1:3
    [~, info] = trainMLP(dataFile, nLayers, 10, 300);
    mlpPerf{nLayers} = info.perf;
end

figure('Name','Fig 4(a) MLP training loss');

subplot(1,2,1); hold on; grid on;
for nLayers = 1:3
    semilogy(mlpPerf{nLayers}, 'DisplayName', sprintf('FC layers = %d', nLayers));
end
xlabel('Epoch'); ylabel('Training Loss (log scale)'); legend show;
title('Fig. 4(a) full range, log scale');

subplot(1,2,2); hold on; grid on;
% Data-driven zoom: look at the tail (after the initial LM crash) so
% the y-limits actually contain data, instead of a hardcoded [0 0.8]
% that would be empty at our loss scale.
tailStart = 10;   % skip the first few epochs' steep drop
zoomMax = 0;
for nLayers = 1:3
    p = mlpPerf{nLayers};
    plot(p, 'DisplayName', sprintf('FC layers = %d', nLayers));
    if numel(p) > tailStart
        zoomMax = max(zoomMax, max(p(tailStart:end)));
    end
end
xlabel('Epoch'); ylabel('Training Loss'); legend show;
if zoomMax > 0
    ylim([0, 1.2*zoomMax]);
end
title('Fig. 4(a) zoomed on post-convergence tail');

%% (b) GRU: sweep number of GRU layers
mlpNet = trainMLP(dataFile, 2, 10, 300);   % fixed MLP for the GRU sweep

figure('Name','Fig 4(b) GRU training loss');

subplot(1,2,1); hold on; grid on;
subplot(1,2,2); hold on; grid on;

for nGRU = 1:3
    [~, trainInfo] = trainGRU(dataFile, mlpNet, nGRU, 100, 40);
    rawLoss = trainInfo.TrainingLoss;

    subplot(1,2,1);
    plot(rawLoss, 'DisplayName', sprintf('GRU layer = %d', nGRU));

    % --- convert per-iteration loss to per-epoch average ---
    maxEpochs = 40;
    itersPerEpoch = round(numel(rawLoss) / maxEpochs);
    if itersPerEpoch < 1, itersPerEpoch = 1; end
    nFullEpochs = floor(numel(rawLoss) / itersPerEpoch);
    epochLoss = mean(reshape(rawLoss(1:nFullEpochs*itersPerEpoch), ...
                              itersPerEpoch, nFullEpochs), 1);

    subplot(1,2,2);
    plot(1:nFullEpochs, epochLoss, '-o', ...
        'DisplayName', sprintf('GRU layer = %d', nGRU));
end

subplot(1,2,1);
xlabel('Iteration'); ylabel('Training Loss'); legend show;
title('Fig. 4(b) raw per-iteration loss (noisy, for reference)');

subplot(1,2,2);
xlabel('Epoch'); ylabel('Training Loss (per-epoch mean)'); legend show;
title('Fig. 4(b) style: per-epoch averaged loss');
