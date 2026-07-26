%% run_pooling_vs_original.m
% Produces the actual comparison artifacts for the paper's REAL claim:
% does the pooling-net detector (buildPooledContextFeatures.m /
% trainPoolingNet.m) beat our reproduced ORIGINAL Deep LAS and the
% classical baselines -- not just the GRU it replaced.
%
% Background: run_direction1_ablation.m's H2 ablation showed the
% correlated-vs-shuffled GRU gap is noise (order doesn't matter), while
% correlated/shuffled both clearly beat i.i.d. pooling (correlation
% CONTENT does matter). run_pooling_vs_gru.m then confirmed a
% permutation-invariant pooling net beats the GRU outright, with a
% widening gap at high SNR where the GRU visibly floors out. This script
% is the next required step: that GRU comparison alone doesn't tell you
% whether the pooling net is actually better than the thing Direction 1
% was supposed to improve on in the first place.
%
% Same structure/metrics as run_direction1_vs_original.m (see that file's
% header + README.md's "What we compare" section for full rationale) --
% only the "ours" arm is swapped from the GRU to the pooling net.

clear; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

% ---- Path to the original codebase ----
% EDIT THIS LINE to match your setup (same as the other run_*.m scripts).
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
ber_mmse        = simulateBER('mmse-hard',    M, SNRdB_range, nBlocksMin);
ber_convLAS     = simulateBER('convlas-hard', M, SNRdB_range, nBlocksMin);
ber_origDeepLAS = simulateBER('deeplas', M, SNRdB_range, nBlocksMin, mlpNet_orig, gruNet_orig);

%% --- Our pooling-net Deep LAS ---
% IMPORTANT: poolNet expects the FIXED-SIZE pooled feature vector from
% buildPooledContextFeatures.m (target features + mean/max/std over the
% correlated context), NOT a raw sequence and NOT the original
% deepLASPredict.m's single-timestep input. Evaluate it ONLY through
% simulateBERPooling.m, which reconstructs that exact representation at
% test time -- same reasoning as simulateBERDirection1.m for the GRU.
cachedNet = fullfile(thisDir, 'result', 'best_pooling_net.mat');
tau_default = 6;
if isfile(cachedNet)
    fprintf('\nLoading cached pooling net from %s\n', cachedNet);
    loaded = load(cachedNet);
    poolNet_d1 = loaded.poolNet;
    mlpNet_d1 = loaded.mlpNet;
    N = loaded.N;
    L = loaded.L;
    Lseq = loaded.Lseq;
    tau = loaded.tau;
else
    fprintf('\nNo cached pooling net found -- run run_pooling_vs_gru.m first,\n');
    fprintf('or training a fresh one now with default settings.\n');
    N = 256; L = 4; tau = tau_default; nRealizations = 200; maxEpochs = 60; stride = 8;
    Bc = coherenceBandwidth(N, L);
    Lseq = round(Bc);
    poolData = buildPooledContextFeatures(M, SNRdB_range, Lseq, N, L, tau, nRealizations, mlpNet_orig, stride);
    poolNet_d1 = trainPoolingNet(poolData, 128, maxEpochs);
    mlpNet_d1 = mlpNet_orig;
end

ber_pooling = simulateBERPooling(M, SNRdB_range, nBlocksMin, mlpNet_d1, poolNet_d1, Lseq, N, L, tau);

%% --- Figure: BER vs SNR comparison ---
figure('Name', 'Pooling net vs. Original: BER comparison');
semilogy(SNRdB_range, ber_mmse, ':x', 'DisplayName', 'MMSE (hard, sanity)'); hold on; grid on;
semilogy(SNRdB_range, ber_convLAS, '-o', 'DisplayName', 'Conv. LAS (hard)');
semilogy(SNRdB_range, ber_origDeepLAS, '-d', 'DisplayName', 'Original Deep LAS (our reproduction)', 'LineWidth', 1.5);
semilogy(SNRdB_range, ber_pooling, '-p', 'Color', [0.13 0.55 0.13], ...
    'DisplayName', 'Pooling-net Deep LAS (ours)', 'LineWidth', 1.5);

yline(1e-4, '--k', 'BER=10^{-4} (paper''s Deep-LAS-vs-Conv-LAS target)', 'LabelHorizontalAlignment','left');
yline(1e-5, ':k',  'BER=10^{-5} (paper''s Deep-LAS-vs-SD target)', 'LabelHorizontalAlignment','left');

set(gca, 'YScale', 'log'); ylim([1e-5 1]);
xlabel('SNR [dB]'); ylabel('Bit Error Rate'); legend('Location','southwest');
title(sprintf('%d-QAM: Original vs. pooling-net Deep LAS vs. classical baselines', M));

%% --- SNR-gain table (the paper's own metric) ---
snr_convLAS_at1e4     = snrAtTargetBER(SNRdB_range, ber_convLAS, 1e-4);
snr_origDeepLAS_at1e4 = snrAtTargetBER(SNRdB_range, ber_origDeepLAS, 1e-4);
snr_pooling_at1e4     = snrAtTargetBER(SNRdB_range, ber_pooling, 1e-4);

gain_orig_vs_convLAS = snr_convLAS_at1e4 - snr_origDeepLAS_at1e4;   % dB, positive = improvement
gain_pool_vs_convLAS = snr_convLAS_at1e4 - snr_pooling_at1e4;
gain_pool_vs_orig    = snr_origDeepLAS_at1e4 - snr_pooling_at1e4;

fprintf('\n================= SNR-GAIN SUMMARY (BER=1e-4 target) =================\n');
fprintf('%-45s %10s\n', 'Quantity', 'Value (dB)');
fprintf('%-45s %10.2f\n', 'Paper: Deep LAS gain vs Conv. LAS', ref.deepLAS_vs_convLAS_gainDB(modIdx));
fprintf('%-45s %10.2f\n', 'Ours: Original Deep LAS gain vs Conv. LAS (reproduction)', gain_orig_vs_convLAS);
fprintf('%-45s %10.2f\n', 'Ours: Pooling net gain vs Conv. LAS', gain_pool_vs_convLAS);
fprintf('%-45s %10.2f\n', 'Ours: Pooling net gain vs Original Deep LAS (the novelty claim)', gain_pool_vs_orig);
fprintf('========================================================================\n');

fprintf('\nInterpretation:\n');
fprintf(' - Row 1 vs Row 2: sanity check -- is our reproduction in the same ballpark as the paper''s claim?\n');
fprintf(' - Row 4: the actual novelty result -- positive means the pooling net beats our own reproduced baseline.\n');
fprintf(' - Unlike Direction 1''s GRU, the pooling net''s advantage over the GRU (see run_pooling_vs_gru.m)\n');
fprintf('   WIDENED at high SNR rather than shrinking -- if Row 4 is also positive and grows with SNR,\n');
fprintf('   that is a much stronger and more paper-worthy result than the GRU ever produced.\n');

resultsTable = table( ...
    {'Paper: DeepLAS vs ConvLAS'; 'Ours: OrigDeepLAS vs ConvLAS'; 'Ours: PoolingNet vs ConvLAS'; 'Ours: PoolingNet vs OrigDeepLAS'}, ...
    [ref.deepLAS_vs_convLAS_gainDB(modIdx); gain_orig_vs_convLAS; gain_pool_vs_convLAS; gain_pool_vs_orig], ...
    'VariableNames', {'Quantity', 'SNRgainDB'});
writetable(resultsTable, fullfile(thisDir, 'result', 'pooling_snr_gain_comparison.csv'));
fprintf('\nSaved comparison table to result/pooling_snr_gain_comparison.csv\n');

save(fullfile(thisDir, 'result', 'pooling_vs_original_curves.mat'), ...
    'SNRdB_range', 'ber_mmse', 'ber_convLAS', 'ber_origDeepLAS', 'ber_pooling');
