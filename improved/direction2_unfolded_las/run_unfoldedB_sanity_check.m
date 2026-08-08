%% run_unfoldedB_sanity_check.m
% CORRECTNESS CHECK, not a results run: with every learnable step size
% fixed at alpha=1 (NO training), unfoldedSoftLAS_B.m should reproduce
% classical softOutputLAS.m almost exactly -- alpha=1 for every layer is
% specifically designed to replicate one classical sweep's behavior (see
% unfoldedLASCore.m's header). If BER differs substantially here, that's
% a bug in the unfolded reimplementation itself, NOT evidence about
% whether training helps or whether depth is insufficient -- fix this
% BEFORE drawing any conclusion from a trained run.
%
% K1/K2 set from checkClassicalSweepDepth.m's measured max sweep counts
% (~4 for Algorithm 1, ~5 for Algorithm 2 across 0-14 dB), with a small
% margin, so depth is not a confound here either.

clear; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

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
K1 = 5;   % measured max was 4, small margin
K2 = 6;   % measured max was 5, small margin

fprintf('--- Classical (fixed, no learning) softOutputLAS ---\n');
ber_classical = simulateBERClassicalSoftLAS(M, SNRdB_range, nBlocksMin);

fprintf('\n--- Unfolded-B with alpha=1 for every layer (should match classical closely) ---\n');
paramsIdentity.alphas1 = ones(K1, 1);
paramsIdentity.alphas2 = ones(K2, 1);
ber_unfoldedB_identity = simulateBERUnfoldedLAS('B', M, SNRdB_range, nBlocksMin, paramsIdentity);

figure('Name', 'Unfolded-B correctness sanity check (alpha=1, no training)');
semilogy(SNRdB_range, ber_classical, '-d', 'Color', [0.93 0.69 0.13], 'LineWidth', 1.5, 'DisplayName', 'Classical softOutputLAS'); hold on; grid on;
semilogy(SNRdB_range, ber_unfoldedB_identity, '-^', 'Color', [0.85 0.10 0.10], 'LineWidth', 1.5, 'DisplayName', 'Unfolded-B, alpha=1 (should match)');
set(gca, 'YScale', 'log'); xlabel('SNR [dB]'); ylabel('Bit Error Rate'); legend('Location', 'southwest');
title(sprintf('%d-QAM: Unfolded-B correctness check (alpha=1 vs. classical)', M));

fprintf('\n================= VERDICT =================\n');
relDiff = abs(ber_classical - ber_unfoldedB_identity) ./ ber_classical;
fprintf('%-6s %-14s %-18s %-12s\n', 'SNR', 'Classical', 'Unfolded(a=1)', 'Rel. diff');
for k = 1:numel(SNRdB_range)
    fprintf('%-6d %-14.4f %-18.4f %-12.1f%%\n', SNRdB_range(k), ber_classical(k), ber_unfoldedB_identity(k), 100*relDiff(k));
end
fprintf('=============================================\n');
fprintf('If rel. diff is consistently small (roughly <20-30%%, within normal Monte\n');
fprintf('Carlo noise for this many blocks), the reimplementation is CORRECT --\n');
fprintf('go back to run_direction2_vs_original.m and the depth/training results\n');
fprintf('there are trustworthy. If it''s consistently large, there IS a bug in\n');
fprintf('unfoldedLASCore.m/unfoldedModLASCounter.m/unfoldedSoftLAS_B.m to find\n');
fprintf('before running any more training experiments.\n');
