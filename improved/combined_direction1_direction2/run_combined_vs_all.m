%% run_combined_vs_all.m
% THE combination step from the original technical plan: "feed the
% structured-sequence input into the unfolded architecture, and show the
% combined system against: (a) the original paper's Deep LAS, (b) your
% Direction-1-only variant, (c) your Direction-2-only variant."
%
% Design (frozen components from both directions, per the confirmed
% decision -- faster, lower risk than joint end-to-end retraining):
%   - Direction 1's pooling net (poolNet) and flat MLP (mlpNet) are
%     loaded FROZEN from direction1_structured_gru/result/best_pooling_net.mat
%     (the confirmed, stability-validated uniform pooling arm).
%   - Direction 2's Option A (Unfolded-B-Soft) is now trained TWICE here,
%     each with the EXACT SAME validated recipe as
%     run_direction2_soft_vs_hard.m / run_direction2_stability_analysis.m,
%     but on two different initial-estimate distributions: once on plain
%     single-subcarrier hard MMSE init (for arm c, matching what's
%     already validated standalone), and once on the FROZEN pooling
%     net's context-informed hard-decided init (for arm d, the actual
%     combined system). This fixes a train/test mismatch found in the
%     first version of this script (arm d was evaluated on an
%     initialization distribution its temperatures were never trained
%     on) -- see buildCombinedTrainingData.m for the full explanation.
%     Expect roughly 2x the single-direction training time as a result.
%   - The NEW piece is simulateBERCombinedFourArms.m: it maps (b)'s
%     hard-decided pooled LLR back to PAM levels as a context-informed
%     initial estimate, and feeds THAT into (c)'s frozen unfolded search
%     instead of the plain single-subcarrier hard MMSE estimate.
%
% Requires original/deep-las-matlab, direction1_structured_gru/, and
% direction2_unfolded_las/ ALL on the MATLAB path.

clear; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
runTag = datestr(now, 'yyyymmdd_HHMMSS'); %#ok<TNOW1,DATST>
fprintf('Run tag: %s\n', runTag);
resultDir = fullfile(thisDir, 'result');
if ~isfolder(resultDir), mkdir(resultDir); end

% ---- EDIT THESE THREE PATHS for your machine ----
origDir = 'D:\Study\3rd year\6th Semester\TW\Original\deep_las_matlab\deep_las_matlab';
dir1Path = 'D:\Study\3rd year\6th Semester\TW\improved\direction1_structured_gru';
dir2Path = 'D:\Study\3rd year\6th Semester\TW\improved\direction2_unfolded_las';

for p = {origDir, dir1Path, dir2Path}
    if isfolder(p{1}), addpath(p{1}); end
end
if exist('genChannel', 'file') ~= 2
    error('Cannot find original/deep-las-matlab on the path -- edit origDir above.');
end
if exist('buildPooledContextFeatures', 'file') ~= 2
    error('Cannot find direction1_structured_gru/ on the path -- edit dir1Path above.');
end
if exist('unfoldedSoftLAS_B_Soft', 'file') ~= 2
    error('Cannot find direction2_unfolded_las/ on the path -- edit dir2Path above.');
end

M = 4;
SNRdB_range = 0:2:20;  % was 0:2:14 -- extended to include 16-20 dB, Direction 1's
                        % CONFIRMED pooling-wins zone (see run_stability_analysis.m).
                        % The first two combined runs stayed entirely inside pooling's
                        % losing-or-tied zone (0-14 dB) and got progressively worse
                        % toward 14 dB -- consistent with feeding the unfolded search a
                        % starting point that is itself worse than the plain estimate in
                        % that range. This tests the direct prediction: if that diagnosis
                        % is right, the combined system should stop losing (or start
                        % winning) specifically above ~16 dB. Direction 2's Option A has
                        % NOT been previously validated above 14 dB -- treat that part as
                        % a new, not-yet-independently-confirmed regime too.
nBlocksMin = 500;

%% --- Load Direction 1's FROZEN pooling net + MLP + channel params ---
cachedPooling = fullfile(dir1Path, 'result', 'best_pooling_net.mat');
if ~isfile(cachedPooling)
    error(['Cannot find %s.\nRun direction1_structured_gru/run_pooling_vs_original.m ' ...
           '(or run_distance_pooling_vs_original.m) first to produce this cache.'], cachedPooling);
end
loadedD1 = load(cachedPooling);
poolNet = loadedD1.poolNet;
mlpNet = loadedD1.mlpNet;
N = loadedD1.N; L = loadedD1.L; Lseq = loadedD1.Lseq; tau = loadedD1.tau;
fprintf('Loaded frozen Direction 1 pooling net (N=%d, L=%d, Lseq=%d, tau=%d)\n', N, L, Lseq, tau);

%% --- Original paper's Deep LAS baseline (arm a) ---
% Trained on the same flat MLP loaded above, matching
% run_direction1_vs_original.m's convention exactly.
dataFileFlat = fullfile(origDir, sprintf('train_%dQAM.mat', M));
if ~isfile(dataFileFlat)
    generateTrainingData(M, 0:2:14, 3000, dataFileFlat);
end
gruNet_orig = trainGRU(dataFileFlat, mlpNet, 2, 100, 40);

%% --- Direction 2's Option A: TWO separate parameter sets ---
% Both use the exact validated recipe/hyperparameters -- only the
% training data's initial-estimate distribution differs, matching what
% each arm actually needs at test time (see buildCombinedTrainingData.m
% for why this fixed the first combined run's mismatch).
K1 = 5; K2 = 6;
d2TrainSamples = 1200; d2MaxEpochs = 6; d2LearnRate = 1e-3; d2MiniBatch = 32; d2InitTemp = 0.5;

fprintf('\n--- Training Direction 2 Option A: PLAIN init (for arm c) ---\n');
trainDataPlain = buildUnfoldedTrainingData(M, SNRdB_range, d2TrainSamples);
unfoldedParamsPlain = trainUnfoldedLASSoft('B', trainDataPlain, K1, K2, d2MaxEpochs, d2LearnRate, d2MiniBatch, d2InitTemp);

fprintf('\n--- Training Direction 2 Option A: POOLING-INFORMED init (for arm d) ---\n');
trainDataCombined = buildCombinedTrainingData(M, SNRdB_range, d2TrainSamples, mlpNet, poolNet, Lseq, N, L, tau);
unfoldedParamsCombined = trainUnfoldedLASSoft('B', trainDataCombined, K1, K2, d2MaxEpochs, d2LearnRate, d2MiniBatch, d2InitTemp);

%% --- The four-way ablation ---
fprintf('\n--- Running combined four-way evaluation ---\n');
BER = simulateBERCombinedFourArms(M, SNRdB_range, nBlocksMin, mlpNet, gruNet_orig, poolNet, ...
    Lseq, N, L, tau, unfoldedParamsPlain, unfoldedParamsCombined);

%% --- Figure ---
figure('Name', 'Combined Direction 1 + Direction 2 vs. all baselines');
semilogy(SNRdB_range, BER.a, '-d', 'Color', [0.93 0.69 0.13], 'DisplayName', 'Original Deep LAS'); hold on; grid on;
semilogy(SNRdB_range, BER.b, '-s', 'Color', [0.13 0.55 0.13], 'DisplayName', 'Direction-1-only (pooling)');
semilogy(SNRdB_range, BER.c, '-^', 'Color', [0.10 0.30 0.85], 'DisplayName', 'Direction-2-only (unfolded)');
semilogy(SNRdB_range, BER.d, '-p', 'Color', [0.85 0.10 0.10], 'DisplayName', 'COMBINED (D1 context -> D2 unfolded)', 'LineWidth', 1.8);

set(gca, 'YScale', 'log'); xlabel('SNR [dB]'); ylabel('Bit Error Rate'); legend('Location', 'southwest');
title(sprintf('%d-QAM: combined system vs. each component alone vs. original', M));

fprintf('\nInterpretation:\n');
fprintf(' - Compare (d) COMBINED against (c) Direction-2-only specifically -- that\n');
fprintf('   isolates whether the pooling-informed initial estimate actually helped\n');
fprintf('   the unfolded search, independent of everything else.\n');
fprintf(' - Compare (d) against (a) Original for the overall "two contributions\n');
fprintf('   together beat the baseline" claim.\n');
fprintf(' - SAME WARNING AS EVERYWHERE ELSE IN THIS PROJECT: this is ONE run. If (c)\n');
fprintf(' - and (d) are close, do not report a winner without the multi-repeat\n');
fprintf('   stability treatment first (adapt run_direction2_stability_analysis.m).\n');

save(fullfile(resultDir, sprintf('combined_vs_all_%s.mat', runTag)), ...
    'SNRdB_range', 'BER', 'unfoldedParamsPlain', 'unfoldedParamsCombined', 'K1', 'K2', 'N', 'L', 'Lseq', 'tau');
fprintf('\nSaved results to result/combined_vs_all_%s.mat\n', runTag);
