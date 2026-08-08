%% run_pooling_vs_gru.m
% Direct A/B test: does a permutation-invariant pooling network match or
% beat the correlated GRU, on the SAME data/SNR range/training budget?
%
% Motivation (see buildPooledContextFeatures.m's header for the full
% story): run_direction1_ablation.m's Part A showed correlated vs.
% shuffled GRU BER is statistically indistinguishable (gap flips sign
% across the SNR sweep), while both clearly beat i.i.d. pooling. That
% means the GRU isn't using context ORDER, just the fact that the
% context is correlated at all -- so a network built to be explicitly
% permutation-invariant (mean/max/std pooling over the context) should
% match or beat the GRU, while being far cheaper/more stable to train
% (no BPTT over 64 timesteps).
%
% Requires original/deep-las-matlab on the MATLAB path (same as
% run_direction1_ablation.m).

clear; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

% ---- Path to the original codebase ----
% EDIT THIS LINE to match your setup (same as the other run_*.m scripts).
originalCodePath = 'D:\Study\3rd year\6th Semester\TW\Original\deep_las_matlab\deep_las_matlab';
if isfolder(originalCodePath)
    addpath(originalCodePath);
end
if exist('genChannel', 'file') ~= 2
    error(['Cannot find the original codebase on the MATLAB path (genChannel.m ' ...
           'was not found).\nEdit the ''originalCodePath'' variable near the top ' ...
           'of this script (currently: %s).'], originalCodePath);
end

M = 4;
N = 256;
nRealizations = 200;
maxEpochs = 60;
SNRdB_range = 0:2:12;
nBlocksMin = 500;
L = 4;
tau = 6;
stride = 8;

Bc = coherenceBandwidth(N, L);
Lseq = round(Bc);
fprintf('N=%d, L=%d -> Bc=%.1f, using Lseq=%d\n', N, L, Bc, Lseq);

% --- shared MLP block (same as the other scripts) ---
dataFileFlat = fullfile(originalCodePath, 'train_4QAM.mat');
if ~isfile(dataFileFlat)
    generateTrainingData(M, 0:2:14, 3000, dataFileFlat);
end
mlpNet = trainMLP(dataFileFlat, 2, 10, 300);

%% --- Correlated GRU: reuse the ablation cache if it matches this config,
% otherwise train fresh, so the comparison is apples-to-apples ---
cachedNet = fullfile(thisDir, 'result', 'best_direction1_gruNet.mat');
gruTrained = false;
if isfile(cachedNet)
    loaded = load(cachedNet);
    if isequal(loaded.N, N) && isequal(loaded.L_A, L) && isequal(loaded.Lseq_A, Lseq)
        fprintf('Reusing cached correlated GRU from %s (matches this config).\n', cachedNet);
        gruCorr = loaded.gruCorr;
        gruTrained = true;
    else
        fprintf('Cached GRU config does not match (N/L/Lseq differ) -- training fresh.\n');
    end
end
if ~gruTrained
    [seqCorr, ~] = buildCorrelatedAndShuffledSequences(M, SNRdB_range, Lseq, N, L, tau, nRealizations, mlpNet, stride);
    gruCorr = trainGRUFromSequences(seqCorr, 2, 100, maxEpochs);
end

%% --- Pooling net: same data budget, permutation-invariant representation ---
poolData = buildPooledContextFeatures(M, SNRdB_range, Lseq, N, L, tau, nRealizations, mlpNet, stride);
poolNet = trainPoolingNet(poolData, 128, maxEpochs);

%% --- Evaluate both on identical BER simulators ---
fprintf('\n=== Evaluating correlated GRU ===\n');
ber_gru  = simulateBERDirection1(M, SNRdB_range, nBlocksMin, mlpNet, gruCorr, Lseq, N, L, tau, [], [], 'correlated');
fprintf('\n=== Evaluating pooling net ===\n');
ber_pool = simulateBERPooling(M, SNRdB_range, nBlocksMin, mlpNet, poolNet, Lseq, N, L, tau);

figure('Name', 'Pooling net vs. correlated GRU');
semilogy(SNRdB_range, ber_gru,  '-*', 'Color', [0.49 0.18 0.56], 'DisplayName', 'Correlated GRU (Direction 1)'); hold on; grid on;
semilogy(SNRdB_range, ber_pool, '-d', 'Color', [0.13 0.55 0.13], 'DisplayName', 'Pooling net (permutation-invariant)');
set(gca, 'YScale', 'log'); xlabel('SNR [dB]'); ylabel('Bit Error Rate'); legend show;
title(sprintf('Pooling net vs. GRU: Lseq=%d (~=Bc), L=%d taps', Lseq, L));

fprintf('\n--- Verdict ---\n');
fprintf('If ber_pool <= ber_gru across most/all of the SNR range, that confirms\n');
fprintf('the useful signal really is order-invariant correlation content, and the\n');
fprintf('simpler/cheaper pooling net is the better architecture to build on.\n');
fprintf('If ber_pool is clearly worse, the GRU may be capturing something the\n');
fprintf('mean/max/std summary loses -- worth trying richer pooling (e.g. more\n');
fprintf('percentiles, or attention-weighted pooling) before reverting to the GRU.\n');

save(fullfile(thisDir, 'result', 'pooling_vs_gru_results.mat'), ...
    'ber_gru', 'ber_pool', 'SNRdB_range', 'Lseq', 'N', 'L', 'tau');
fprintf('\nSaved results to result/pooling_vs_gru_results.mat\n');

save(fullfile(thisDir, 'result', 'best_pooling_net.mat'), ...
    'poolNet', 'mlpNet', 'N', 'L', 'Lseq', 'tau');
fprintf('Saved the trained pooling net to result/best_pooling_net.mat for reuse.\n');
