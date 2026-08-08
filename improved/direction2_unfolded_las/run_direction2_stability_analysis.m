%% run_direction2_stability_analysis.m
% Multi-repeat confirmation for Option A's result, same discipline as
% ../direction1_structured_gru/run_stability_analysis.m: retrains
% Unfolded-B-Soft from scratch (fresh channels, fresh training data,
% fresh initialization) across nRuns independent repeats, and reports
% mean +/- std BER plus a per-SNR significance check against classical
% softOutputLAS, instead of trusting any single run.
%
% WHY THIS IS NEEDED NOW: two quick-mode runs looked promising (green
% beats gold at nearly every SNR, gap widening at high SNR), but one of
% those two runs was of uncertain independence (looked suspiciously
% identical to an earlier run), and two runs was never the bar this
% project set for a headline claim -- Direction 1's equivalent result
% only became trustworthy after 5 independent repeats revealed the real
% (more nuanced) three-regime picture. Do not write up Option A as a
% clean win until this script confirms it.
%
% Uses quickMode-scale settings (nTrainSamples=1200, maxEpochs=6) so
% nRuns repeats stays feasible in total time (~1 hour x nRuns) -- this
% trades some per-run training quality for enough independent samples to
% assess real variability, same tradeoff Direction 1 made.

clear; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
runTag = datestr(now, 'yyyymmdd_HHMMSS'); %#ok<TNOW1,DATST>
fprintf('Stability-analysis run tag: %s\n', runTag);

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
nRuns = 4;   % ~1 hour each at quickMode scale -- start here, bump if time allows
M = 4;
SNRdB_range = 0:2:14;
nBlocksMin = 500;
K1 = 5; K2 = 6;
nTrainSamples = 1200;   % quickMode scale, matches the runs already done
maxEpochs = 6;
learnRate = 1e-3;
miniBatchSize = 32;
initTemp = 0.5;

nSNR = numel(SNRdB_range);
ber_classical_all = zeros(nRuns, nSNR);
ber_soft_all = zeros(nRuns, nSNR);

for r = 1:nRuns
    fprintf('\n========================= REPEAT %d / %d =========================\n', r, nRuns);

    % Classical baseline re-evaluated fresh every repeat too (its own
    % Monte Carlo blocks are freshly drawn each time -- this is why you
    % saw a small wobble in the classical curve between runs already).
    ber_classical_all(r, :) = simulateBERClassicalSoftLAS(M, SNRdB_range, nBlocksMin);

    trainData = buildUnfoldedTrainingData(M, SNRdB_range, nTrainSamples);
    paramsBSoft = trainUnfoldedLASSoft('B', trainData, K1, K2, maxEpochs, learnRate, miniBatchSize, initTemp);
    ber_soft_all(r, :) = simulateBERUnfoldedLASSoft('B', M, SNRdB_range, nBlocksMin, paramsBSoft);

    fprintf('Repeat %d done. Classical@14dB=%.5f, Soft@14dB=%.5f\n', ...
        r, ber_classical_all(r,end), ber_soft_all(r,end));
end

%% --- Aggregate ---
mean_classical = mean(ber_classical_all, 1); std_classical = std(ber_classical_all, 0, 1);
mean_soft = mean(ber_soft_all, 1);           std_soft = std(ber_soft_all, 0, 1);

%% --- Plot: mean +/- 1 std shaded bands ---
figure('Name', 'Direction 2 stability analysis: Option A vs. classical');
hold on; grid on;
plotBandLog(SNRdB_range, mean_classical, std_classical, [0.93 0.69 0.13], 'Classical softOutputLAS');
plotBandLog(SNRdB_range, mean_soft, std_soft, [0.10 0.60 0.10], 'Unfolded-B-Soft (Option A)');
set(gca, 'YScale', 'log'); xlabel('SNR [dB]'); ylabel('Bit Error Rate'); legend('Location', 'southwest');
title(sprintf('%d-QAM: mean +/- 1 std BER across %d independent repeats', M, nRuns));

%% --- Per-SNR significance check ---
fprintf('\n================= PER-SNR SIGNIFICANCE CHECK =================\n');
fprintf('%-6s %-14s %-14s %-10s %-s\n', 'SNR', 'Classical', 'Soft(mean)', 'z', 'Verdict');
for k = 1:nSNR
    se_c = std_classical(k) / sqrt(nRuns);
    se_s = std_soft(k) / sqrt(nRuns);
    z = (mean_classical(k) - mean_soft(k)) / sqrt(se_c^2 + se_s^2 + eps);
    if z > 2
        verdict = 'Option A significantly BETTER';
    elseif z < -2
        verdict = 'Classical significantly better';
    elseif abs(z) < 1
        verdict = 'noise (cannot tell)';
    else
        verdict = 'ambiguous';
    end
    fprintf('%-6d %-14.5f %-14.5f %-10.2f %-s\n', SNRdB_range(k), mean_classical(k), mean_soft(k), z, verdict);
end
fprintf('================================================================\n');
fprintf('z > +2 at MOST/ALL SNR points, especially consistently at high SNR, is what\n');
fprintf('would confirm Option A as a real, reportable win -- not just 1-2 good runs.\n');

save(fullfile(thisDir, 'result', sprintf('direction2_stability_%s.mat', runTag)), ...
    'SNRdB_range', 'nRuns', 'ber_classical_all', 'ber_soft_all', ...
    'mean_classical', 'std_classical', 'mean_soft', 'std_soft');
fprintf('\nSaved results to result/direction2_stability_%s.mat\n', runTag);

function plotBandLog(x, m, s, color, name)
lower = max(m - s, m*1e-3);
upper = m + s;
fill([x, fliplr(x)], [lower, fliplr(upper)], color, ...
    'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
semilogy(x, m, '-o', 'Color', color, 'LineWidth', 1.5, 'DisplayName', name);
end
