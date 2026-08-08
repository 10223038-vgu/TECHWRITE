%% run_lowsnr_regularization_test.m
% Tests the OVERFITTING/CAPACITY hypothesis for why the pooling nets
% significantly underperform Original Deep LAS below ~12 dB SNR (see
% run_stability_analysis.m's results): the pooling net (2x128 hidden
% units, dropout=0.05, default L2=1e-4) is far larger and more weakly
% regularized than Original's MLP (10 units) + GRU (100 units) combined,
% and low-SNR training targets (true LLRs) are themselves noisiest --
% exactly the regime where a bigger, weakly-regularized net is most
% prone to fitting noise instead of signal.
%
% Trains TWO variants of the uniform pooling net, both from
% buildPooledContextFeatures.m's identical dataset, differing ONLY in
% capacity/regularization:
%   - 'baseline':    numHiddenUnits=128, dropout=0.05, L2=1e-4 (current
%                    settings, i.e. what run_stability_analysis.m used)
%   - 'regularized': numHiddenUnits=64,  dropout=0.15, L2=1e-3 (smaller,
%                    more regularized)
% and compares both against Original Deep LAS, using nRuns independent
% repeats (same reasoning as run_stability_analysis.m: single runs are
% not trustworthy for these comparisons) but restricted to the LOW-SNR
% range (0:2:12) where the gap actually showed up, to keep this
% experiment faster than a full 0-20 dB stability run.
%
% VERDICT: if 'regularized' closes most/all of the gap to Original, the
% capacity/overfitting hypothesis is confirmed and the fix is to just use
% these settings going forward. If it doesn't move the needle, the
% low-SNR gap is more likely intrinsic to the pooled-feature
% representation itself (see run_distance_pooling_vs_original.m's next
% steps in THEORY_POOLING.md) rather than a training-recipe problem.

clear; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
runTag = datestr(now, 'yyyymmdd_HHMMSS'); %#ok<TNOW1,DATST>
fprintf('Run tag: %s\n', runTag);

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
nRuns = 3;   % start here; bump to 5 if time allows, same reasoning as run_stability_analysis.m
M = 4;
SNRdB_range = 0:2:12;   % LOW-SNR ONLY -- this is where the gap to Original showed up
nBlocksMin = 1000;
ber_minErrors = 100;
ber_maxBlocksFactor = 2000;

N = 256; L = 4; tau = 6; nRealizations = 400; maxEpochs = 60; stride = 8;
Bc = coherenceBandwidth(N, L);
Lseq = round(Bc);

dataFileFlat = fullfile(origDir, sprintf('train_%dQAM_snr%dto%d.mat', M, SNRdB_range(1), SNRdB_range(end)));

nSNR = numel(SNRdB_range);
ber_orig_all       = zeros(nRuns, nSNR);
ber_baseline_all    = zeros(nRuns, nSNR);
ber_regularized_all = zeros(nRuns, nSNR);

for r = 1:nRuns
    fprintf('\n========================= REPEAT %d / %d =========================\n', r, nRuns);

    if ~isfile(dataFileFlat)
        generateTrainingData(M, SNRdB_range, 3000, dataFileFlat);
    end
    mlpNet = trainMLP(dataFileFlat, 2, 10, 300);
    gruNet_orig = trainGRU(dataFileFlat, mlpNet, 2, 100, 40);
    ber_orig_all(r, :) = simulateBER('deeplas', M, SNRdB_range, nBlocksMin, mlpNet, gruNet_orig);

    % Both variants trained on the EXACT SAME dataset draw for this
    % repeat, so any difference between them is attributable only to
    % capacity/regularization, not to a different random data sample.
    poolData = buildPooledContextFeatures(M, SNRdB_range, Lseq, N, L, tau, nRealizations, mlpNet, stride);

    poolNetBaseline = trainPoolingNet(poolData, 128, maxEpochs, 0.05, 1e-4);
    ber_baseline_all(r, :) = simulateBERPooling(M, SNRdB_range, nBlocksMin, mlpNet, poolNetBaseline, Lseq, N, L, tau, ber_minErrors, ber_maxBlocksFactor);

    poolNetRegularized = trainPoolingNet(poolData, 64, maxEpochs, 0.15, 1e-3);
    ber_regularized_all(r, :) = simulateBERPooling(M, SNRdB_range, nBlocksMin, mlpNet, poolNetRegularized, Lseq, N, L, tau, ber_minErrors, ber_maxBlocksFactor);

    fprintf('Repeat %d: Orig@0dB=%.4f, Baseline@0dB=%.4f, Regularized@0dB=%.4f\n', ...
        r, ber_orig_all(r,1), ber_baseline_all(r,1), ber_regularized_all(r,1));
end

%% --- Aggregate ---
mean_orig = mean(ber_orig_all, 1);              std_orig = std(ber_orig_all, 0, 1);
mean_base = mean(ber_baseline_all, 1);           std_base = std(ber_baseline_all, 0, 1);
mean_reg  = mean(ber_regularized_all, 1);        std_reg  = std(ber_regularized_all, 0, 1);

%% --- Plot ---
figure('Name', 'Low-SNR regularization test');
semilogy(SNRdB_range, mean_orig, '-d', 'Color', [0.93 0.69 0.13], 'LineWidth', 1.5, 'DisplayName', 'Original Deep LAS'); hold on; grid on;
semilogy(SNRdB_range, mean_base, '-s', 'Color', [0.13 0.55 0.13], 'LineWidth', 1.5, 'DisplayName', 'Pooling: baseline (128u, drop.05, L2 1e-4)');
semilogy(SNRdB_range, mean_reg,  '-^', 'Color', [0.10 0.30 0.85], 'LineWidth', 1.5, 'DisplayName', 'Pooling: regularized (64u, drop.15, L2 1e-3)');
set(gca, 'YScale', 'log'); xlabel('SNR [dB]'); ylabel('Bit Error Rate'); legend('Location', 'northeast');
title(sprintf('%d-QAM: does regularization close the low-SNR gap? (mean over %d repeats)', M, nRuns));

%% --- Significance check: does 'regularized' close the gap to Original? ---
fprintf('\n================= LOW-SNR REGULARIZATION TEST =================\n');
fprintf('%-6s %-12s %-12s %-12s %-10s %-10s %-s\n', 'SNR', 'Orig', 'Baseline', 'Regularized', 'z(Base-O)', 'z(Reg-O)', 'Verdict');
for k = 1:nSNR
    se_o = std_orig(k)/sqrt(nRuns);
    se_b = std_base(k)/sqrt(nRuns);
    se_r = std_reg(k)/sqrt(nRuns);
    z_base = (mean_orig(k) - mean_base(k)) / sqrt(se_o^2 + se_b^2 + eps);
    z_reg  = (mean_orig(k) - mean_reg(k))  / sqrt(se_o^2 + se_r^2 + eps);

    if abs(z_reg) < 1
        verdict = 'GAP CLOSED (regularized ~= Original)';
    elseif z_reg > z_base + 1
        verdict = 'improved, but gap remains';
    else
        verdict = 'no meaningful improvement';
    end
    fprintf('%-6d %-12.4f %-12.4f %-12.4f %-10.2f %-10.2f %-s\n', ...
        SNRdB_range(k), mean_orig(k), mean_base(k), mean_reg(k), z_base, z_reg, verdict);
end
fprintf('=================================================================\n');
fprintf('Reminder: z is (Original mean - arm mean)/combined SE. Negative z = arm is WORSE than Original.\n');
fprintf('If most rows say GAP CLOSED, use trainPoolingNet(poolData, 64, maxEpochs, 0.15, 1e-3) going forward.\n');
fprintf('If not, the low-SNR gap is likely intrinsic to the pooled-feature representation, not a training-recipe fix.\n');

save(fullfile(thisDir, 'result', sprintf('lowsnr_regularization_test_%s.mat', runTag)), ...
    'SNRdB_range', 'nRuns', 'ber_orig_all', 'ber_baseline_all', 'ber_regularized_all', ...
    'mean_orig', 'std_orig', 'mean_base', 'std_base', 'mean_reg', 'std_reg');
fprintf('\nSaved results to result/lowsnr_regularization_test_%s.mat\n', runTag);
