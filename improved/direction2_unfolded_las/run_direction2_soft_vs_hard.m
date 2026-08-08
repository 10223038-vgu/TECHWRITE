    %% run_direction2_soft_vs_hard.m
% THE Option A result: trains Unfolded-B-Soft (fully differentiable
% softmax relaxation, see THEORY_UNFOLDING.md Section 3 and
% unfoldedLASCoreSoft.m) end-to-end against true bits, and compares
% against classical (fixed) softOutputLAS and Option B's best result
% (learned step size + detached hard decision), which plateaued at/near
% classical performance without a clear win.
%
% Learning rate starts conservative (1e-3) from the start, given Option
% B's lesson that 1e-2 caused unstable, non-converging training for the
% harder (Algorithm 1 + Algorithm 2) variant.
%
% IMPORTANT: run_unfoldedB_soft_sanity_check.m must pass BEFORE trusting
% anything from this script -- it validates the new soft-relaxation code
% is correct, independent of whether training helps.

clear; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
resultDir = fullfile(thisDir, 'result');
if ~isfolder(resultDir)
    mkdir(resultDir);
end
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

M = 4;
SNRdB_range = 0:2:14;
nBlocksMin = 500;
K1 = 5; K2 = 6;
% Option A's fully-differentiable graph is meaningfully more expensive
% per sample than Option B's (see THEORY_UNFOLDING.md) -- the earlier
% nTrainSamples=8000/maxEpochs=25 config was still too slow in practice.
% quickMode below gets a FAST first real signal (not a tuned final
% result) so you know whether this is worth waiting longer for at all,
% before committing real time to a bigger run.
quickMode = false;
if quickMode
    nTrainSamples = 1200;
    maxEpochs = 6;
else
    nTrainSamples = 8000;
    maxEpochs = 25;
end
learnRate = 1e-3;
miniBatchSize = 32;
initTemp = 0.5;

fprintf('K1=%d, K2=%d, nTrainSamples=%d, maxEpochs=%d, learnRate=%.4f, initTemp=%.2f\n', ...
    K1, K2, nTrainSamples, maxEpochs, learnRate, initTemp);

%% --- Baselines ---
fprintf('\n--- Classical baselines ---\n');
ber_mmse = simulateBER('mmse-hard', M, SNRdB_range, nBlocksMin);
ber_convLAS = simulateBER('convlas-hard', M, SNRdB_range, nBlocksMin);
ber_classicalLAS = simulateBERClassicalSoftLAS(M, SNRdB_range, nBlocksMin);

%% --- Training data ---
fprintf('\n--- Building training data ---\n');
trainData = buildUnfoldedTrainingData(M, SNRdB_range, nTrainSamples);

%% --- Unfolded-B-Soft ---
fprintf('\n--- Training Unfolded-B-Soft (both algorithms, soft relaxation) ---\n');
fprintf('Watch the printed T1/T2 values each epoch -- if they shrink toward 0,\n');
fprintf('training is sharpening toward a hard decision (i.e. toward classical\n');
fprintf('behavior); if they grow, it is deliberately blending multiple candidates\n');
fprintf('more than the classical algorithm ever would.\n');
paramsBSoft = trainUnfoldedLASSoft('B', trainData, K1, K2, maxEpochs, learnRate, miniBatchSize, initTemp);

fprintf('\n--- Evaluating Unfolded-B-Soft ---\n');
ber_unfoldedBSoft = simulateBERUnfoldedLASSoft('B', M, SNRdB_range, nBlocksMin, paramsBSoft);

%% --- Figure ---
figure('Name', 'Direction 2: soft relaxation (Option A) vs. classical baselines');
semilogy(SNRdB_range, ber_mmse, ':x', 'DisplayName', 'MMSE (hard, sanity)'); hold on; grid on;
semilogy(SNRdB_range, ber_convLAS, '-o', 'DisplayName', 'Conv. LAS (hard)');
semilogy(SNRdB_range, ber_classicalLAS, '-d', 'Color', [0.93 0.69 0.13], 'DisplayName', 'Classical softOutputLAS (fixed, no learning)', 'LineWidth', 1.5);
semilogy(SNRdB_range, ber_unfoldedBSoft, '-p', 'Color', [0.10 0.60 0.10], 'DisplayName', 'Unfolded-B-Soft (Option A, learned temperatures)', 'LineWidth', 1.5);

set(gca, 'YScale', 'log'); xlabel('SNR [dB]'); ylabel('Bit Error Rate'); legend('Location', 'southwest');
title(sprintf('%d-QAM: Option A (soft relaxation) vs. classical baselines', M));

fprintf('\nInterpretation:\n');
fprintf(' - Compare against "Classical softOutputLAS" directly, same as Option B.\n');
fprintf(' - If this beats classical where Option B (learned step size only) could\n');
fprintf('   not, that confirms the diagnosis in THEORY_UNFOLDING.md -- a fully\n');
fprintf('   differentiable candidate selection was the missing lever.\n');
fprintf(' - Same warning as always: do not trust a single run for a close call.\n');
fprintf('   Use run_stability_analysis.m''s multi-repeat approach as a template\n');
fprintf('   before reporting a winner if this looks close.\n');

save(fullfile(thisDir, 'result', sprintf('direction2_soft_vs_hard_%s.mat', runTag)), ...
    'SNRdB_range', 'ber_mmse', 'ber_convLAS', 'ber_classicalLAS', 'ber_unfoldedBSoft', ...
    'paramsBSoft', 'K1', 'K2', 'initTemp', 'learnRate');
fprintf('\nSaved results to result/direction2_soft_vs_hard_%s.mat\n', runTag);
