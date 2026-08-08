%% run_unfoldedB_soft_sanity_check.m
% CORRECTNESS CHECK for the new Option A (soft relaxation) code, same
% role as run_unfoldedB_sanity_check.m played for Option B: as
% temperature -> 0, unfoldedLASCoreSoft.m's softmax sharpens toward a
% one-hot at the minimum-cost candidate, which is EXACTLY the classical
% hard decision. So a very low (fixed, UNtrained) temperature should
% reproduce classical softOutputLAS.m closely. If it doesn't, there's a
% bug in the new soft-relaxation code (unfoldedLASCoreSoft.m /
% unfoldedLAS1Soft.m / unfoldedModLASCounterSoft.m /
% unfoldedSoftLAS_B_Soft.m) to find BEFORE running any training.

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
K1 = 5; K2 = 6;   % same as the Option B sanity check's measured depth

fprintf('--- Classical (fixed, no learning) softOutputLAS ---\n');
ber_classical = simulateBERClassicalSoftLAS(M, SNRdB_range, nBlocksMin);

fprintf('\n--- Unfolded-B-Soft with very low fixed temperature (T=0.01, no training) ---\n');
paramsLowTemp.temps1 = 0.01 * ones(K1, 1);
paramsLowTemp.temps2 = 0.01 * ones(K2, 1);
ber_soft_lowtemp = simulateBERUnfoldedLASSoft('B', M, SNRdB_range, nBlocksMin, paramsLowTemp);

figure('Name', 'Unfolded-B-Soft correctness sanity check (low temperature, no training)');
semilogy(SNRdB_range, ber_classical, '-d', 'Color', [0.93 0.69 0.13], 'LineWidth', 1.5, 'DisplayName', 'Classical softOutputLAS'); hold on; grid on;
semilogy(SNRdB_range, ber_soft_lowtemp, '-^', 'Color', [0.10 0.30 0.85], 'LineWidth', 1.5, 'DisplayName', 'Unfolded-B-Soft, T=0.01 (should match)');
set(gca, 'YScale', 'log'); xlabel('SNR [dB]'); ylabel('Bit Error Rate'); legend('Location', 'southwest');
title(sprintf('%d-QAM: Unfolded-B-Soft correctness check (low temperature vs. classical)', M));

fprintf('\n================= VERDICT =================\n');
relDiff = abs(ber_classical - ber_soft_lowtemp) ./ ber_classical;
fprintf('%-6s %-14s %-18s %-12s\n', 'SNR', 'Classical', 'Soft(T=0.01)', 'Rel. diff');
for k = 1:numel(SNRdB_range)
    fprintf('%-6d %-14.4f %-18.4f %-12.1f%%\n', SNRdB_range(k), ber_classical(k), ber_soft_lowtemp(k), 100*relDiff(k));
end
fprintf('=============================================\n');
fprintf('If rel. diff is consistently small (normal Monte Carlo range), the soft\n');
fprintf('relaxation code is correct -- proceed to training with confidence. If not,\n');
fprintf('there is a bug in the new soft-relaxation files to find first.\n');
