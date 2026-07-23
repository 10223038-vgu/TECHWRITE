%% run_fig10_modelBasedComparison.m
% Reproduces Fig. 10: BER vs SNR comparing the proposed detectors
% against other model-based/data-driven baselines, for both ZF and
% MMSE initialization, plus DetNet and the optimal soft-output SD.
%
% Baselines included:
%   Conv. ZF/MMSE LAS (hard)   - las1Hard.m via simulateBER('convlas-hard',...)
%   MLP only                   - trainMLP.m via simulateBER('mlp-only',...)
%   Prop. ZF/MMSE Soft-LAS     - softOutputLAS.m via simulateBER('softlas',...)
%   Prop. Deep LAS             - trainMLP.m + trainGRU.m via simulateBER('deeplas',...)
%   DetNet                     - trainDetNet.m / detNetPredict.m (EXPERIMENTAL, see README)
%   SD (optimal)                - bruteForceSD.m (exact ML/MAP via exhaustive search, Nt=4 only)
%
% NOTE ON DETNET: this is the most experimental component of the whole
% package (custom dlarray training loop, not the standard trainNetwork
% API -- see the big warning at the top of trainDetNet.m). If it
% errors out, comment out the DetNet block below and run the rest of
% the figure without it.

clear; clc;
cfg = getConfig();
SNRdB_range = 0:2:14;
nBlocksMin = 500;   % adaptive stopping (see simulateBER.m) will run more as needed

includeDetNet = true;   % set false to skip the experimental DetNet baseline

figure('Name', 'Fig. 10: model-based/data-driven detector comparison');

for panelIdx = 1:numel(cfg.M_list)
    M = cfg.M_list(panelIdx);
    fprintf('\n=== Fig. 10, M = %d-QAM ===\n', M);

    dataFile = sprintf('train_%dQAM.mat', M);
    if ~isfile(dataFile)
        generateTrainingData(M, 0:2:14, 3000, dataFile);
    end
    mlpNet = trainMLP(dataFile, 2, 10, 300);
    gruNet = trainGRU(dataFile, mlpNet, 2, 100, 40);

    % --- Conventional hard LAS, both initializations ---
    ber_convZF   = simulateBER('convlas-hard', M, SNRdB_range, nBlocksMin, [], [], [], [], 'zf');
    ber_convMMSE = simulateBER('convlas-hard', M, SNRdB_range, nBlocksMin, [], [], [], [], 'mmse');

    % --- Standalone MLP ---
    ber_mlpOnly = simulateBER('mlp-only', M, SNRdB_range, nBlocksMin, mlpNet);

    % --- Proposed two-step Soft-LAS, both initializations ---
    ber_softZF   = simulateBER('softlas', M, SNRdB_range, nBlocksMin, [], [], [], [], 'zf');
    ber_softMMSE = simulateBER('softlas', M, SNRdB_range, nBlocksMin, [], [], [], [], 'mmse');

    % --- Proposed Deep LAS ---
    ber_deepLAS = simulateBER('deeplas', M, SNRdB_range, nBlocksMin, mlpNet, gruNet);

    % --- Optimal soft-output SD (exact brute force, Nt=4 only) ---
    ber_sd = simulateBER('sd-optimal', M, SNRdB_range, nBlocksMin);

    % --- DetNet (experimental) ---
    if includeDetNet
        fprintf('Training DetNet for M=%d (this is the slow, experimental part)...\n', M);
        detNet = trainDetNet(M, 10, 40, 8, 2000, 20, 1e-3);
        ber_detnet = simulateBER('detnet', M, SNRdB_range, nBlocksMin, [], detNet);
    end

    subplot(1, numel(cfg.M_list), panelIdx); hold on; grid on;
    set(gca, 'YScale', 'log');

    semilogy(SNRdB_range, ber_convZF,   '-o', 'DisplayName', 'Conv. ZF LAS-hard');
    semilogy(SNRdB_range, ber_convMMSE, '-o', 'DisplayName', 'Conv. MMSE LAS-hard');
    semilogy(SNRdB_range, ber_mlpOnly,  '-s', 'DisplayName', 'MLP only approx.');
    semilogy(SNRdB_range, ber_softZF,   '-^', 'DisplayName', 'Prop. ZF LAS-soft');
    semilogy(SNRdB_range, ber_softMMSE, '-^', 'DisplayName', 'Prop. MMSE LAS-soft');
    semilogy(SNRdB_range, ber_deepLAS,  '-d', 'DisplayName', 'Prop. Deep LAS-soft', 'LineWidth', 1.5);
    if includeDetNet
        semilogy(SNRdB_range, ber_detnet, '-x', 'DisplayName', 'DetNet');
    end
    semilogy(SNRdB_range, ber_sd, '-k*', 'DisplayName', 'SD (optimal)', 'LineWidth', 1.5);

    xlabel('SNR [dB]'); ylabel('Bit Error Rate'); legend('Location','southwest');
    ylim([1e-5 1]);
    title(sprintf('%d-QAM', M));
end

sgtitle('Fig. 10: Proposed detectors vs. model-based/data-driven baselines');
