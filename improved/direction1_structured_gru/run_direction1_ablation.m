%% run_direction1_ablation.m
% Runs the three ablation experiments from THEORY.md (Section 1.7-1.8):
%   Part A (H2, the core proof): correlated vs. shuffled vs. i.i.d.
%     sequences at a fixed Lseq = coherence bandwidth. If correlated beats
%     shuffled despite IDENTICAL content (see buildCorrelatedAndShuffledSequences.m),
%     that proves the GRU is exploiting real correlation structure.
%   Part B (H1): sweep Lseq relative to the coherence bandwidth Bc, at
%     fixed L (taps). Expect BER to improve as Lseq -> Bc and degrade for
%     Lseq << Bc or Lseq >> Bc.
%   Part C (H3): sweep L (taps) at FIXED Lseq. Expect the correlated-vs-
%     shuffled BER gap to shrink as L increases (Bc shrinks, so a
%     fixed-length window captures progressively less correlation).
%
% Requires original/deep-las-matlab on the MATLAB path (this script adds
% it automatically, assuming the folder layout in README.md).
%
% NOTE: default sizes here are deliberately small (few realizations, few
% training epochs) so you can smoke-test the whole pipeline quickly before
% committing to a full-scale run -- see the "SCALE UP FOR REAL RESULTS"
% comments below.

clear; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

% ---- Path to the original codebase ----
% EDIT THIS LINE to point at wherever your original/deep-las-matlab (or
% equivalent) folder actually lives -- folder names/casing vary depending
% on how things were unzipped, so this is left explicit rather than
% guessed via a relative path.
originalCodePath = 'D:\Study\3rd year\6th Semester\TW\Original\deep_las_matlab\deep_las_matlab';
if isfolder(originalCodePath)
    addpath(originalCodePath);
end
if exist('genChannel', 'file') ~= 2
    error(['Cannot find the original codebase on the MATLAB path (genChannel.m ' ...
           'was not found).\nEdit the ''originalCodePath'' variable near the top ' ...
           'of this script (currently: %s) to point at the folder containing ' ...
           'genChannel.m, qamHelpers.m, trainMLP.m, etc.'], originalCodePath);
end

M = 4;
N = 256;              % subcarriers per channel realization
nRealizations = 20;   % SCALE UP FOR REAL RESULTS (e.g. 200+)
maxEpochs = 20;        % SCALE UP FOR REAL RESULTS (e.g. 40-100)
SNRdB_range = 0:2:12;
nBlocksMin = 300;

% --- shared MLP block: reuse the ORIGINAL flat i.i.d.-trained MLP.
% Valid without retraining because each H_k is still marginally CN(0,1)
% regardless of L/tau -- see genMultipathChannel.m's header comment.
dataFileFlat = fullfile(originalCodePath, 'train_4QAM.mat');
if ~isfile(dataFileFlat)
    generateTrainingData(M, 0:2:14, 3000, dataFileFlat);
end
mlpNet = trainMLP(dataFileFlat, 2, 10, 300);

results = struct();

%% ---------------- Part A: H2 (the core proof) ----------------
fprintf('\n=== Part A: correlated vs shuffled vs i.i.d. (H2) ===\n');
L_A = 4;
Bc_A = coherenceBandwidth(N, L_A);
Lseq_A = round(Bc_A);
fprintf('N=%d, L=%d -> Bc=%.1f, using Lseq=%d\n', N, L_A, Bc_A, Lseq_A);

snrForTraining = 8;   % train at a representative SNR
[seqCorr, seqShuf] = buildCorrelatedAndShuffledSequences(M, snrForTraining, Lseq_A, N, L_A, 6, nRealizations, mlpNet);
seqIID = buildIIDSequences(M, snrForTraining, Lseq_A, numel(seqCorr.X), mlpNet);

gruCorr = trainGRUFromSequences(seqCorr, 2, 100, maxEpochs);
gruShuf = trainGRUFromSequences(seqShuf, 2, 100, maxEpochs);
gruIID  = trainGRUFromSequences(seqIID,  2, 100, maxEpochs);

ber_corr = simulateBER('deeplas', M, SNRdB_range, nBlocksMin, mlpNet, gruCorr);
ber_shuf = simulateBER('deeplas', M, SNRdB_range, nBlocksMin, mlpNet, gruShuf);
ber_iid  = simulateBER('deeplas', M, SNRdB_range, nBlocksMin, mlpNet, gruIID);

figure('Name', 'Part A: H2 ablation (correlated vs shuffled vs iid)');
semilogy(SNRdB_range, ber_corr, '-o', 'DisplayName', 'Correlated (proposed)'); hold on; grid on;
semilogy(SNRdB_range, ber_shuf, '-s', 'DisplayName', 'Shuffled control (same content)');
semilogy(SNRdB_range, ber_iid,  '-^', 'DisplayName', 'Original i.i.d. pooling');
set(gca, 'YScale', 'log'); xlabel('SNR [dB]'); ylabel('BER'); legend show;
title(sprintf('H2 ablation: Lseq=%d (~=Bc), L=%d taps', Lseq_A, L_A));

results.partA = struct('ber_corr', ber_corr, 'ber_shuf', ber_shuf, 'ber_iid', ber_iid, ...
    'Lseq', Lseq_A, 'L', L_A, 'SNRdB_range', SNRdB_range);

save(fullfile(thisDir, 'result', 'best_direction1_gruNet.mat'), 'gruCorr', 'mlpNet', 'Lseq_A', 'L_A', 'N');
fprintf('Saved the Part A correlated GRU net to result/best_direction1_gruNet.mat for reuse.\n');

fprintf('\n--- H2 verdict ---\n');
fprintf('If ber_corr is consistently below ber_shuf despite identical content,\n');
fprintf('that is direct evidence the GRU exploits correlation order, not just capacity.\n');

%% ---------------- Part B: H1 (Lseq sweep) ----------------
fprintf('\n=== Part B: Lseq sweep relative to Bc (H1) ===\n');
L_B = 4;
Bc_B = coherenceBandwidth(N, L_B);
LseqFactors = [0.25, 0.5, 1, 2, 4];
ber_vs_Lseq = zeros(numel(LseqFactors), numel(SNRdB_range));

for fi = 1:numel(LseqFactors)
    Lseq_i = max(2, round(LseqFactors(fi) * Bc_B));
    if Lseq_i > N, continue; end
    fprintf('Lseq = %d (%.2fx Bc=%.1f)\n', Lseq_i, LseqFactors(fi), Bc_B);
    [seqCorr_i, ~] = buildCorrelatedAndShuffledSequences(M, snrForTraining, Lseq_i, N, L_B, 6, nRealizations, mlpNet);
    gru_i = trainGRUFromSequences(seqCorr_i, 2, 100, maxEpochs);
    ber_vs_Lseq(fi, :) = simulateBER('deeplas', M, SNRdB_range, nBlocksMin, mlpNet, gru_i);
end

figure('Name', 'Part B: H1 ablation (Lseq sweep)');
hold on; grid on; set(gca, 'YScale', 'log');
for fi = 1:numel(LseqFactors)
    if any(ber_vs_Lseq(fi,:) > 0)
        semilogy(SNRdB_range, ber_vs_Lseq(fi,:), '-o', ...
            'DisplayName', sprintf('Lseq = %.2fx Bc', LseqFactors(fi)));
    end
end
xlabel('SNR [dB]'); ylabel('BER'); legend show;
title(sprintf('H1 ablation: Lseq swept relative to Bc=%.1f (L=%d)', Bc_B, L_B));

results.partB = struct('ber_vs_Lseq', ber_vs_Lseq, 'LseqFactors', LseqFactors, ...
    'Bc', Bc_B, 'L', L_B, 'SNRdB_range', SNRdB_range);

%% ---------------- Part C: H3 (L sweep at fixed Lseq) ----------------
fprintf('\n=== Part C: L sweep at fixed Lseq (H3) ===\n');
Lseq_C = 32;   % fixed sequence length
L_values = [2, 4, 8, 16];
gap_at_referenceSNR = zeros(size(L_values));
refSNRidx = find(SNRdB_range == 8, 1);
if isempty(refSNRidx), refSNRidx = round(numel(SNRdB_range)/2); end

for li = 1:numel(L_values)
    L_i = L_values(li);
    fprintf('L = %d taps (Bc = %.1f, fixed Lseq = %d)\n', L_i, coherenceBandwidth(N, L_i), Lseq_C);
    [seqCorr_i, seqShuf_i] = buildCorrelatedAndShuffledSequences(M, snrForTraining, Lseq_C, N, L_i, 6, nRealizations, mlpNet);
    gruCorr_i = trainGRUFromSequences(seqCorr_i, 2, 100, maxEpochs);
    gruShuf_i = trainGRUFromSequences(seqShuf_i, 2, 100, maxEpochs);

    ber_corr_i = simulateBER('deeplas', M, SNRdB_range, nBlocksMin, mlpNet, gruCorr_i);
    ber_shuf_i = simulateBER('deeplas', M, SNRdB_range, nBlocksMin, mlpNet, gruShuf_i);

    gap_at_referenceSNR(li) = ber_shuf_i(refSNRidx) - ber_corr_i(refSNRidx);
end

figure('Name', 'Part C: H3 ablation (L sweep)');
plot(L_values, gap_at_referenceSNR, '-o'); grid on;
xlabel('Number of channel taps L'); ylabel('BER gap (shuffled - correlated)');
title(sprintf('H3 ablation: correlated-vs-shuffled gap at SNR=%d dB, fixed Lseq=%d', ...
    SNRdB_range(refSNRidx), Lseq_C));

results.partC = struct('gap_at_referenceSNR', gap_at_referenceSNR, 'L_values', L_values, ...
    'Lseq', Lseq_C, 'refSNRdB', SNRdB_range(refSNRidx));

fprintf('\n--- H3 verdict ---\n');
fprintf('If the gap shrinks as L increases, that confirms coherence bandwidth\n');
fprintf('(not some other confound) is driving the Part A/B results.\n');

save(fullfile(thisDir, 'result', 'ablation_results.mat'), 'results');
fprintf('\nSaved all ablation results to result/ablation_results.mat\n');
