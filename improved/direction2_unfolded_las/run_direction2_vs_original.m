%% run_direction2_vs_original.m
% First real Direction 2 result: trains BOTH unfolded-LAS variants
% (Unfolded-A: Algorithm 1 only + learned readout; Unfolded-B: both
% algorithms, classical combination formula, both step-size sets
% learnable) end-to-end against TRUE TRANSMITTED BITS, and compares
% against MMSE, Conv. LAS (hard), and the CLASSICAL (non-learned)
% softOutputLAS.m two-step detector.
%
% STANDALONE per the confirmed plan: i.i.d. flat channel model, matching
% the original codebase exactly -- no correlated/pooled context from
% Direction 1 yet. Combining the two directions is a later step, once
% each works independently.
%
% Requires original/deep-las-matlab on the MATLAB path.

clear; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
resultDir = fullfile(thisDir, 'result');
if ~isfolder(resultDir)
    mkdir(resultDir);   % was previously crashing the final save() after all
                         % the (expensive) training already ran -- fixed
end
runTag = datestr(now, 'yyyymmdd_HHMMSS'); %#ok<TNOW1,DATST>
fprintf('Run tag: %s\n', runTag);

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

M = 4;
SNRdB_range = 0:2:14;
nBlocksMin = 500;

% --- Unfolded-network config ---
% K1/K2 set from checkClassicalSweepDepth.m's MEASURED convergence depth
% (max ~4 sweeps for Algorithm 1, ~5 for Algorithm 2 across 0-14 dB),
% plus a small margin -- not a guess. IMPORTANT: before trusting any
% result from this script, run run_unfoldedB_sanity_check.m first (fixed
% alpha=1, no training) and confirm it reproduces classical
% softOutputLAS.m's BER closely. If it doesn't, that's a reimplementation
% bug to fix here, not a training/depth question this script can answer.
K1 = 5;
K2 = 6;
nTrainSamples = 16000;   % was 4000 -- that was a correctness check, not a real attempt
maxEpochs = 40;          % was 15
learnRate = 1e-2;
miniBatchSize = 32;

fprintf(['NOTE: the previous run (nTrainSamples=4000, maxEpochs=15) was a\n' ...
    'correctness check, not a tuned attempt -- Unfolded-B tracked close to\n' ...
    'classical softOutputLAS but did not clearly beat it. Scaled up training\n' ...
    'here to see whether that gap (or lack of one) holds with a real training\n' ...
    'budget before concluding anything.\n']);

fprintf('K1=%d, K2=%d, nTrainSamples=%d, maxEpochs=%d\n', K1, K2, nTrainSamples, maxEpochs);
fprintf(['NOTE: this trains with a per-sample forward-pass loop (not\n' ...
    'vectorized across a batch) -- expect this to take a while. Start\n' ...
    'with these modest sizes to confirm everything runs end-to-end before\n' ...
    'scaling up nTrainSamples/maxEpochs.\n']);

%% --- Baselines: MMSE, Conv. LAS, classical (non-learned) softOutputLAS ---
fprintf('\n--- Classical baselines ---\n');
ber_mmse    = simulateBER('mmse-hard',    M, SNRdB_range, nBlocksMin);
ber_convLAS = simulateBER('convlas-hard', M, SNRdB_range, nBlocksMin);

% Classical softOutputLAS.m as its own baseline curve (NOT the original
% paper's GRU -- this isolates "does learning the step sizes help
% relative to the classical algorithm itself," which is Direction 2's
% actual question, separate from Direction 1's MLP-GRU baseline).
ber_classicalLAS = simulateBERClassicalSoftLAS(M, SNRdB_range, nBlocksMin);

%% --- Training data (standalone, i.i.d. flat channel) ---
fprintf('\n--- Building training data ---\n');
trainData = buildUnfoldedTrainingData(M, SNRdB_range, nTrainSamples);

%% --- Unfolded-A ---
fprintf('\n--- Training Unfolded-A (Algorithm 1 only + learned readout) ---\n');
paramsA = trainUnfoldedLAS('A', trainData, K1, K2, maxEpochs, learnRate, miniBatchSize);
fprintf('\n--- Evaluating Unfolded-A ---\n');
ber_unfoldedA = simulateBERUnfoldedLAS('A', M, SNRdB_range, nBlocksMin, paramsA);

%% --- Unfolded-B ---
fprintf('\n--- Training Unfolded-B (Algorithm 1 + Algorithm 2, classical combination) ---\n');
paramsB = trainUnfoldedLAS('B', trainData, K1, K2, maxEpochs, learnRate, miniBatchSize);
fprintf('\n--- Evaluating Unfolded-B ---\n');
ber_unfoldedB = simulateBERUnfoldedLAS('B', M, SNRdB_range, nBlocksMin, paramsB);

%% --- Figure ---
figure('Name', 'Direction 2: Unfolded LAS vs. classical baselines');
semilogy(SNRdB_range, ber_mmse, ':x', 'DisplayName', 'MMSE (hard, sanity)'); hold on; grid on;
semilogy(SNRdB_range, ber_convLAS, '-o', 'DisplayName', 'Conv. LAS (hard)');
semilogy(SNRdB_range, ber_classicalLAS, '-d', 'DisplayName', 'Classical softOutputLAS (fixed, no learning)', 'LineWidth', 1.5);
semilogy(SNRdB_range, ber_unfoldedA, '-s', 'Color', [0.10 0.30 0.85], 'DisplayName', 'Unfolded-A (Alg.1 only + learned readout)', 'LineWidth', 1.5);
semilogy(SNRdB_range, ber_unfoldedB, '-^', 'Color', [0.85 0.10 0.10], 'DisplayName', 'Unfolded-B (Alg.1+Alg.2, learned step sizes)', 'LineWidth', 1.5);

set(gca, 'YScale', 'log'); xlabel('SNR [dB]'); ylabel('Bit Error Rate'); legend('Location', 'southwest');
title(sprintf('%d-QAM: Direction 2 (unfolded LAS) vs. classical baselines', M));

fprintf('\nInterpretation:\n');
fprintf(' - Compare Unfolded-A/B against "Classical softOutputLAS" specifically --\n');
fprintf('   that comparison isolates whether LEARNING the step sizes helps over the\n');
fprintf('   fixed classical algorithm, independent of Direction 1''s MLP-GRU question.\n');
fprintf(' - As with Direction 1, do NOT trust a single run for close calls -- if\n');
fprintf('   Unfolded-A/B and Classical LAS end up close together, this needs the same\n');
fprintf('   multi-repeat stability treatment as run_stability_analysis.m before\n');
fprintf('   reporting a winner.\n');

save(fullfile(thisDir, 'result', sprintf('direction2_vs_original_%s.mat', runTag)), ...
    'SNRdB_range', 'ber_mmse', 'ber_convLAS', 'ber_classicalLAS', 'ber_unfoldedA', 'ber_unfoldedB', ...
    'paramsA', 'paramsB', 'K1', 'K2');
fprintf('\nSaved results to result/direction2_vs_original_%s.mat\n', runTag);
