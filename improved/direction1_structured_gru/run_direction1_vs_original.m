%% run_direction1_vs_original.m
% Produces the actual comparison artifacts for the paper: BER curves
% overlaying our reproduced original Deep LAS, our Direction 1
% (structured-GRU) Deep LAS, and the classical baselines, PLUS a summary
% table of "SNR needed to reach BER=1e-4 / 1e-5" -- the exact metric the
% original paper itself reports its headline numbers in (see
% paperReferenceNumbers.m), so every row is directly comparable.
%
% METRICS COMPARED (see README.md's "What we compare" section for the
% full rationale):
%   1. BER vs SNR curves (Conv. LAS, MMSE, Original Deep LAS, Direction 1)
%   2. SNR required to reach BER=1e-4 (paper's Deep-LAS-vs-Conv-LAS metric)
%   3. SNR required to reach BER=1e-5 (paper's Deep-LAS-vs-SD metric)
%   4. SNR GAIN relative to Conv. LAS at each BER target (this is the
%      literal quantity the paper reports as "2.55 dB / 3 dB gain")
%
% We do NOT claim to overlay the paper's raw curves (we don't have their
% digitized data) -- we compare against their REPORTED headline numbers
% (paperReferenceNumbers.m) side by side with our own reproduction and
% our own improvement, so the reader can see: does our reproduction land
% near what the paper claims, and does our improvement move further in
% the same direction.

clear; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

% ---- Path to the original codebase ----
% EDIT THIS LINE to point at wherever your original/deep-las-matlab (or
% equivalent) folder actually lives -- folder names/casing vary depending
% on how things were unzipped, so this is left explicit rather than
% guessed via a relative path.
origDir = 'D:\Study\3rd year\6th Semester\TW\Original\deep_las_matlab\deep_las_matlab';

if isfolder(origDir)
    addpath(origDir);
end
if exist('genChannel', 'file') ~= 2
    error(['Cannot find the original codebase on the MATLAB path (genChannel.m ' ...
           'was not found).\nEdit the ''origDir'' variable near the top of this ' ...
           'script (currently: %s) to point at the folder containing ' ...
           'genChannel.m, qamHelpers.m, trainMLP.m, etc.'], origDir);
end

M = 4;
SNRdB_range = 0:1:14;
nBlocksMin = 500;
ref = paperReferenceNumbers();
modIdx = find(ref.modulationOrders == M, 1);

%% --- Baselines + our reproduced original Deep LAS ---
dataFileFlat = fullfile(origDir, sprintf('train_%dQAM.mat', M));
if ~isfile(dataFileFlat)
    generateTrainingData(M, 0:2:14, 3000, dataFileFlat);
end
mlpNet_orig = trainMLP(dataFileFlat, 2, 10, 300);
gruNet_orig = trainGRU(dataFileFlat, mlpNet_orig, 2, 100, 40);

fprintf('\n--- Baselines & reproduced original Deep LAS ---\n');
ber_mmse       = simulateBER('mmse-hard',    M, SNRdB_range, nBlocksMin);
ber_convLAS    = simulateBER('convlas-hard', M, SNRdB_range, nBlocksMin);
ber_origDeepLAS = simulateBER('deeplas', M, SNRdB_range, nBlocksMin, mlpNet_orig, gruNet_orig);

%% --- Our Direction 1 (structured GRU) Deep LAS ---
% IMPORTANT: gruNet_d1 was trained on multi-timestep (length-Lseq)
% correlated-subcarrier sequences (trainGRUFromSequences.m /
% buildCorrelatedAndShuffledSequences.m). It must be EVALUATED the same
% way. Do NOT pass it to the original simulateBER.m -- that always routes
% through deepLASPredict.m, which builds a SINGLE-timestep input
% regardless of what gruNet expects. That mismatch silently produces
% near-random LLRs (BER ~ 0.5 at every SNR, flat line) with no error, and
% is exactly what the flat purple curve was. simulateBERDirection1.m
% reconstructs the correct Lseq-long correlated context per prediction.
cachedNet = fullfile(thisDir, 'result', 'best_direction1_gruNet.mat');
tau_default = 6;   % must match the tau used to build the training sequences
if isfile(cachedNet)
    fprintf('\nLoading cached Direction 1 GRU from %s\n', cachedNet);
    loaded = load(cachedNet);
    gruNet_d1 = loaded.gruCorr;
    mlpNet_d1 = loaded.mlpNet;
    N = loaded.N;
    L = loaded.L_A;
    Lseq = loaded.Lseq_A;
    if isfield(loaded, 'tau_A')
        tau = loaded.tau_A;
    else
        tau = tau_default;
        fprintf(['(Cached net has no stored tau -- assuming tau=%d, the ' ...
            'default used when this cache was produced by ' ...
            'run_direction1_ablation.m. Re-run the ablation script to ' ...
            'refresh the cache with tau saved explicitly.)\n'], tau);
    end
else
    fprintf('\nNo cached Direction 1 net found -- run run_direction1_ablation.m first,\n');
    fprintf('or training a fresh one now with default settings.\n');
    N = 256; L = 4; tau = tau_default; nRealizations = 40; maxEpochs = 40;
    Bc = coherenceBandwidth(N, L);
    Lseq = round(Bc);
    [seqCorr, ~] = buildCorrelatedAndShuffledSequences(M, 8, Lseq, N, L, tau, nRealizations, mlpNet_orig);
    gruNet_d1 = trainGRUFromSequences(seqCorr, 2, 100, maxEpochs);
    mlpNet_d1 = mlpNet_orig;
end

ber_direction1 = simulateBERDirection1(M, SNRdB_range, nBlocksMin, mlpNet_d1, gruNet_d1, ...
    Lseq, N, L, tau, [], [], 'correlated');

%% --- Figure: BER vs SNR comparison ---
figure('Name', 'Direction 1 vs. Original: BER comparison');
semilogy(SNRdB_range, ber_mmse, ':x', 'DisplayName', 'MMSE (hard, sanity)'); hold on; grid on;
semilogy(SNRdB_range, ber_convLAS, '-o', 'DisplayName', 'Conv. LAS (hard)');
semilogy(SNRdB_range, ber_origDeepLAS, '-d', 'DisplayName', 'Original Deep LAS (our reproduction)', 'LineWidth', 1.5);
semilogy(SNRdB_range, ber_direction1, '-p', 'DisplayName', 'Direction 1: structured-GRU Deep LAS (ours)', 'LineWidth', 1.5);

yline(1e-4, '--k', 'BER=10^{-4} (paper''s Deep-LAS-vs-Conv-LAS target)', 'LabelHorizontalAlignment','left');
yline(1e-5, ':k',  'BER=10^{-5} (paper''s Deep-LAS-vs-SD target)', 'LabelHorizontalAlignment','left');

set(gca, 'YScale', 'log'); ylim([1e-5 1]);
xlabel('SNR [dB]'); ylabel('Bit Error Rate'); legend('Location','southwest');
title(sprintf('%d-QAM: Original vs. Direction 1 vs. classical baselines', M));

%% --- SNR-gain table (the paper's own metric) ---
snr_convLAS_at1e4 = snrAtTargetBER(SNRdB_range, ber_convLAS, 1e-4);
snr_origDeepLAS_at1e4 = snrAtTargetBER(SNRdB_range, ber_origDeepLAS, 1e-4);
snr_direction1_at1e4 = snrAtTargetBER(SNRdB_range, ber_direction1, 1e-4);

gain_orig_vs_convLAS = snr_convLAS_at1e4 - snr_origDeepLAS_at1e4;   % dB, positive = improvement
gain_d1_vs_convLAS   = snr_convLAS_at1e4 - snr_direction1_at1e4;
gain_d1_vs_orig      = snr_origDeepLAS_at1e4 - snr_direction1_at1e4;

fprintf('\n================= SNR-GAIN SUMMARY (BER=1e-4 target) =================\n');
fprintf('%-40s %10s\n', 'Quantity', 'Value (dB)');
fprintf('%-40s %10.2f\n', 'Paper: Deep LAS gain vs Conv. LAS', ref.deepLAS_vs_convLAS_gainDB(modIdx));
fprintf('%-40s %10.2f\n', 'Ours: Original Deep LAS gain vs Conv. LAS (reproduction)', gain_orig_vs_convLAS);
fprintf('%-40s %10.2f\n', 'Ours: Direction 1 gain vs Conv. LAS', gain_d1_vs_convLAS);
fprintf('%-40s %10.2f\n', 'Ours: Direction 1 gain vs Original Deep LAS (the novelty claim)', gain_d1_vs_orig);
fprintf('========================================================================\n');

fprintf('\nInterpretation:\n');
fprintf(' - Row 1 vs Row 2: sanity check -- is our reproduction in the same ballpark as the paper''s claim?\n');
fprintf(' - Row 4: the actual novelty result -- positive means Direction 1 beats our own reproduced baseline.\n');

resultsTable = table( ...
    {'Paper: DeepLAS vs ConvLAS'; 'Ours: OrigDeepLAS vs ConvLAS'; 'Ours: Direction1 vs ConvLAS'; 'Ours: Direction1 vs OrigDeepLAS'}, ...
    [ref.deepLAS_vs_convLAS_gainDB(modIdx); gain_orig_vs_convLAS; gain_d1_vs_convLAS; gain_d1_vs_orig], ...
    'VariableNames', {'Quantity', 'SNRgainDB'});
writetable(resultsTable, fullfile(thisDir, 'result', 'snr_gain_comparison.csv'));
fprintf('\nSaved comparison table to result/snr_gain_comparison.csv\n');

save(fullfile(thisDir, 'result', 'direction1_vs_original_curves.mat'), ...
    'SNRdB_range', 'ber_mmse', 'ber_convLAS', 'ber_origDeepLAS', 'ber_direction1');
