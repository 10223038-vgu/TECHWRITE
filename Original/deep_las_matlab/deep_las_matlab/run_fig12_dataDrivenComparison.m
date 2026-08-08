%% run_fig12_dataDrivenComparison.m
% Reproduces Fig. 12: BER performance comparison of the proposed
% Deep LAS (MLP+GRU hybrid) against standalone data-driven baselines
% -- MLP-only, LSTM, Bi-LSTM, and a standalone GRU (each fed the same
% MLP rough-LLR input via trainSeqModel.m) -- with two subplots,
% (a) 4-QAM and (b) 16-QAM, BER vs SNR. Optionally overlays the
% MMSE-hard classical baseline per your Fig. 12 requirements doc.
%
% NOTE ON SCALE: the paper trains these RNN baselines for 500 epochs
% with 800 hidden units (Section V-C). That is substantially heavier
% than the Deep LAS GRU block itself (100 units, 40 epochs) -- expect
% this script to take a while. Reduce numHiddenUnits/maxEpochs below
% for a faster, lower-fidelity smoke test first.

clear; clc;
cfg = getConfig();
SNRdB_range = 0:2:14;
nBlocks = 1500;
includeMMSEBaseline = true;

% Turn these down for a quick test run, up for a faithful reproduction:
seqHiddenUnits = 800;   % paper: 800
seqEpochs = 500;        % paper: 500

figure('Name', 'Fig. 12: Deep LAS vs data-driven baselines');

for panelIdx = 1:numel(cfg.M_list)
    M = cfg.M_list(panelIdx);
    fprintf('\n=== Fig. 12, M = %d-QAM ===\n', M);

    dataFile = sprintf('train_%dQAM.mat', M);
    if ~isfile(dataFile)
        generateTrainingData(M, 0:2:14, 3000, dataFile);
    end

    % --- Shared MLP block (rough LLR estimate feeding every RNN) ---
    mlpNet = trainMLP(dataFile, 2, 10, 300);

    % --- Proposed Deep LAS (paper's own GRU hyperparameters) ---
    gruNet_deepLAS = trainGRU(dataFile, mlpNet, 2, 100, 40);

    % --- Fig. 12 baselines (paper's benchmark hyperparameters) ---
    lstmNet   = trainSeqModel(dataFile, mlpNet, 'lstm',   seqHiddenUnits, seqEpochs);
    bilstmNet = trainSeqModel(dataFile, mlpNet, 'bilstm', seqHiddenUnits, seqEpochs);
    gruNet_fig12 = trainSeqModel(dataFile, mlpNet, 'gru', seqHiddenUnits, seqEpochs);

    % --- BER curves ---
    ber_mlpOnly = simulateBER('mlp-only', M, SNRdB_range, nBlocks, mlpNet);
    ber_lstm    = simulateBER('deeplas',  M, SNRdB_range, nBlocks, mlpNet, lstmNet);
    ber_bilstm  = simulateBER('deeplas',  M, SNRdB_range, nBlocks, mlpNet, bilstmNet);
    ber_gru     = simulateBER('deeplas',  M, SNRdB_range, nBlocks, mlpNet, gruNet_fig12);
    ber_deepLAS = simulateBER('deeplas',  M, SNRdB_range, nBlocks, mlpNet, gruNet_deepLAS);

    subplot(1, numel(cfg.M_list), panelIdx); hold on; grid on;
    set(gca, 'YScale', 'log');

    if includeMMSEBaseline
        ber_mmse = simulateBER('mmse-hard', M, SNRdB_range, nBlocks);
        semilogy(SNRdB_range, ber_mmse, ':x', 'DisplayName', 'MMSE (hard)');
    end
    semilogy(SNRdB_range, ber_mlpOnly, '-s', 'DisplayName', 'MLP only');
    semilogy(SNRdB_range, ber_lstm,    '-v', 'DisplayName', 'LSTM');
    semilogy(SNRdB_range, ber_bilstm,  '-^', 'DisplayName', 'Bi-LSTM');
    semilogy(SNRdB_range, ber_gru,     '-p', 'DisplayName', 'GRU');
    semilogy(SNRdB_range, ber_deepLAS, '-d', 'DisplayName', 'Prop. Deep LAS', 'LineWidth', 1.5);

    xlabel('SNR [dB]'); ylabel('Bit Error Rate'); legend('Location','southwest');
    ylim([1e-5 1]);
    title(sprintf('(%s) %d-QAM', char('a' + panelIdx - 1), M));
end

sgtitle('Fig. 12: Proposed Deep LAS vs. other data-driven detection techniques');
