%% run_stability_analysis.m
% Repeats the FULL train+evaluate cycle (fresh random channels, fresh
% network initialization, fresh training-set draws every time) multiple
% independent times, and reports MEAN +/- STD BER curves instead of a
% single run's curve.
%
% *** WHY THIS EXISTS ***
% Comparing run_distance_pooling_vs_original.m's output across a couple
% of manual re-runs showed the high-SNR (16-20 dB) ordering between
% Original Deep LAS and the pooling nets FLIPS from run to run -- one
% run had pooling ahead, another had Original ahead, by margins similar
% in size to the flip itself. That means the single-run comparison at
% high SNR is dominated by run-to-run noise (channel realizations,
% training-set shuffling, network weight initialization), not a real,
% repeatable effect, and no amount of eyeballing more individual runs
% resolves that -- you need the actual sampling distribution.
%
% This script retrains everything from scratch each repeat (NOT reusing
% any cached net -- reusing a cache would only capture BER-simulation
% noise, not the training-randomness that's a big part of what's driving
% the flips you observed) and reports, per SNR point:
%   - mean BER and standard deviation across nRuns independent repeats,
%     for MMSE, Conv. LAS, Original Deep LAS, Uniform pooling, and
%     Distance-aware pooling
%   - a shaded-band plot (mean +/- 1 std) for the three "ours"/original
%     detector arms, so the high-SNR region's overlap (or lack of it) is
%     visible directly instead of inferred from single-run crossings
%   - a rough per-SNR significance check (z-like statistic on the
%     difference of means, using each arm's own standard error) between
%     Original and each pooling arm, flagging which SNR points show a
%     difference clearly bigger than combined run-to-run noise
%
% COST WARNING: this is nRuns times the cost of one full
% run_distance_pooling_vs_original.m execution (retraining the flat MLP,
% the original GRU, AND both pooling nets from scratch every repeat).
% Start with a small nRuns (3) and a shallower BER depth to gauge total
% runtime before committing to a long overnight run.

clear; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
runTag = datestr(now, 'yyyymmdd_HHMMSS'); %#ok<TNOW1,DATST>
fprintf('Stability-analysis run tag: %s\n', runTag);

% ---- Path to the original codebase ----
origDir = 'D:\Study\3rd year\6th Semester\TW\Original\deep_las_matlab\deep_las_matlab';
if isfolder(origDir)
    addpath(origDir);
end
if exist('genChannel', 'file') ~= 2
    error(['Cannot find the original codebase on the MATLAB path (genChannel.m ' ...
           'was not found).\nEdit the ''origDir'' variable near the top of this ' ...
           'script (currently: %s).'], origDir);
end

%% --- Config ---
nRuns = 5;   % START SMALLER (e.g. 3) FIRST to gauge total runtime, then increase if feasible
M = 4;
SNRdB_range = 0:2:20;

% Deliberately shallower than run_distance_pooling_vs_original.m's
% deepSim=true setting -- with nRuns repeats, total simulated bits is
% already nRuns x this depth, and what matters here is getting the
% SAMPLING DISTRIBUTION across independent repeats, not maximum depth on
% any single one. Increase if runtime allows.
nBlocksMin = 1000;
ber_minErrors = 100;
ber_maxBlocksFactor = 2000;

N = 256; L = 4; tau = 6; nRealizations = 400; maxEpochs = 60; stride = 8;
Bc = coherenceBandwidth(N, L);
Lseq = round(Bc);
nearFrac = 0.25;
decayScale = Lseq/2;

nSNR = numel(SNRdB_range);
ber_mmse_all    = zeros(nRuns, nSNR);
ber_convLAS_all = zeros(nRuns, nSNR);
ber_orig_all    = zeros(nRuns, nSNR);
ber_uniform_all = zeros(nRuns, nSNR);
ber_dist_all    = zeros(nRuns, nSNR);

dataFileFlat = fullfile(origDir, sprintf('train_%dQAM_snr%dto%d.mat', M, SNRdB_range(1), SNRdB_range(end)));

for r = 1:nRuns
    fprintf('\n========================= REPEAT %d / %d =========================\n', r, nRuns);

    % --- Original Deep LAS: retrained fresh every repeat (NOT cached) ---
    % Reuses the flat MLP training-data FILE if present (that data itself
    % is just samples, not a trained net, and regenerating it every
    % repeat would add cost without adding meaningful variability since
    % it's already a large i.i.d. dataset) but ALWAYS retrains mlpNet and
    % gruNet_orig fresh, since network weight initialization is exactly
    % one of the noise sources this script is trying to capture.
    if ~isfile(dataFileFlat)
        generateTrainingData(M, SNRdB_range, 3000, dataFileFlat);
    end
    mlpNet = trainMLP(dataFileFlat, 2, 10, 300);
    gruNet_orig = trainGRU(dataFileFlat, mlpNet, 2, 100, 40);

    ber_mmse_all(r, :)    = simulateBER('mmse-hard',    M, SNRdB_range, nBlocksMin);
    ber_convLAS_all(r, :) = simulateBER('convlas-hard', M, SNRdB_range, nBlocksMin);
    ber_orig_all(r, :)    = simulateBER('deeplas', M, SNRdB_range, nBlocksMin, mlpNet, gruNet_orig);

    % --- Uniform pooling net: retrained fresh every repeat ---
    poolDataUniform = buildPooledContextFeatures(M, SNRdB_range, Lseq, N, L, tau, nRealizations, mlpNet, stride);
    poolNetUniform = trainPoolingNet(poolDataUniform, 128, maxEpochs);
    ber_uniform_all(r, :) = simulateBERPooling(M, SNRdB_range, nBlocksMin, mlpNet, poolNetUniform, Lseq, N, L, tau, ber_minErrors, ber_maxBlocksFactor);

    % --- Distance-aware pooling net: retrained fresh every repeat ---
    poolDataDist = buildDistanceWeightedPooledFeatures(M, SNRdB_range, Lseq, N, L, tau, nRealizations, mlpNet, stride, nearFrac, decayScale);
    poolNetDist = trainPoolingNet(poolDataDist, 128, maxEpochs);
    ber_dist_all(r, :) = simulateBERDistancePooling(M, SNRdB_range, nBlocksMin, mlpNet, poolNetDist, Lseq, N, L, tau, nearFrac, decayScale, ber_minErrors, ber_maxBlocksFactor);

    fprintf('Repeat %d done. Original@20dB=%.4f, Uniform@20dB=%.4f, Distance@20dB=%.4f\n', ...
        r, ber_orig_all(r,end), ber_uniform_all(r,end), ber_dist_all(r,end));
end

%% --- Aggregate: mean +/- std across repeats ---
mean_mmse = mean(ber_mmse_all, 1);       std_mmse = std(ber_mmse_all, 0, 1);
mean_conv = mean(ber_convLAS_all, 1);    std_conv = std(ber_convLAS_all, 0, 1);
mean_orig = mean(ber_orig_all, 1);       std_orig = std(ber_orig_all, 0, 1);
mean_unif = mean(ber_uniform_all, 1);    std_unif = std(ber_uniform_all, 0, 1);
mean_dist = mean(ber_dist_all, 1);       std_dist = std(ber_dist_all, 0, 1);

%% --- Plot: mean curves with +/-1 std shaded bands (log-scale-safe) ---
figure('Name', 'Stability analysis: mean +/- std across repeats');
hold on; grid on;

plotBandLog(SNRdB_range, mean_mmse, std_mmse, [0.3 0.3 0.9], 'MMSE (hard, sanity)');
plotBandLog(SNRdB_range, mean_conv, std_conv, [0.9 0.4 0.1], 'Conv. LAS (hard)');
plotBandLog(SNRdB_range, mean_orig, std_orig, [0.93 0.69 0.13], 'Original Deep LAS (our reproduction)');
plotBandLog(SNRdB_range, mean_unif, std_unif, [0.13 0.55 0.13], 'Uniform pooling net');
plotBandLog(SNRdB_range, mean_dist, std_dist, [0.85 0.10 0.10], 'Distance-aware pooling net');

set(gca, 'YScale', 'log'); ylim([1e-4 1]);
xlabel('SNR [dB]'); ylabel('Bit Error Rate'); legend('Location', 'southwest');
title(sprintf('%d-QAM: mean +/- 1 std BER across %d independent repeats', M, nRuns));

%% --- Per-SNR significance check: Original vs. each pooling arm ---
% Rough z-like statistic on the difference of means, using each arm's
% own standard error (std/sqrt(nRuns)). This is a heuristic, not a
% rigorous test (nRuns is small, BER is not normally distributed) -- but
% |z|>2 is a reasonable bar for "this difference is probably bigger than
% run-to-run noise," while |z|<1 is a clear "cannot tell them apart from
% these repeats."
fprintf('\n================= PER-SNR SIGNIFICANCE CHECK =================\n');
fprintf('%-6s %-12s %-12s %-12s %-10s %-12s %-10s\n', 'SNR', 'Orig(mean)', 'Unif(mean)', 'Dist(mean)', 'z(U-O)', 'z(D-O)', 'Verdict');
for k = 1:nSNR
    se_orig = std_orig(k) / sqrt(nRuns);
    se_unif = std_unif(k) / sqrt(nRuns);
    se_dist = std_dist(k) / sqrt(nRuns);

    z_unif = (mean_orig(k) - mean_unif(k)) / sqrt(se_orig^2 + se_unif^2 + eps);
    z_dist = (mean_orig(k) - mean_dist(k)) / sqrt(se_orig^2 + se_dist^2 + eps);

    if abs(z_dist) > 2 || abs(z_unif) > 2
        verdict = 'likely real diff';
    elseif abs(z_dist) < 1 && abs(z_unif) < 1
        verdict = 'noise (cannot tell)';
    else
        verdict = 'ambiguous';
    end

    fprintf('%-6d %-12.4f %-12.4f %-12.4f %-10.2f %-12.2f %-10s\n', ...
        SNRdB_range(k), mean_orig(k), mean_unif(k), mean_dist(k), z_unif, z_dist, verdict);
end
fprintf('================================================================\n');
fprintf('z > +2 means pooling net has LOWER mean BER than Original by more than ~2 standard errors.\n');
fprintf('z < -2 means pooling net has HIGHER mean BER than Original by more than ~2 standard errors.\n');
fprintf('|z| < 1 at 16-20 dB is what you''d see if the single-run flip-flopping you observed was pure noise.\n');

save(fullfile(thisDir, 'result', sprintf('stability_analysis_%s.mat', runTag)), ...
    'SNRdB_range', 'nRuns', ...
    'ber_mmse_all', 'ber_convLAS_all', 'ber_orig_all', 'ber_uniform_all', 'ber_dist_all', ...
    'mean_mmse', 'std_mmse', 'mean_conv', 'std_conv', 'mean_orig', 'std_orig', ...
    'mean_unif', 'std_unif', 'mean_dist', 'std_dist');
fprintf('\nSaved all per-run curves + mean/std to result/stability_analysis_%s.mat\n', runTag);

function plotBandLog(x, m, s, color, name)
% Shaded +/-1 std band around a mean curve, clipped at a small positive
% floor so the lower band never goes <=0 (which log-scale can't show).
lower = max(m - s, m*1e-3);
upper = m + s;
fill([x, fliplr(x)], [lower, fliplr(upper)], color, ...
    'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
semilogy(x, m, '-o', 'Color', color, 'LineWidth', 1.5, 'DisplayName', name);
end