%% run_fig9_BERcomparison.m
% Reproduces the structure of Fig. 9: BER vs SNR for the conventional
% hard-output 1-LAS, the proposed model-based two-step soft-output
% LAS, the proposed data-driven Deep LAS, and the standalone MLP.
%
% Run generate + train scripts first (see README.md, Quick Start).

clear; clc;
cfg = getConfig();
SNRdB_range = 0:2:14;
nBlocks = 500;     % raise substantially for smooth, low-BER curves

figure; hold on; set(gca,'YScale','log'); grid on;
colors = lines(8);

for M = cfg.M_list
    fprintf('\n=== M = %d-QAM ===\n', M);

    % --- Train Deep LAS for this modulation order (or load if cached) ---
    dataFile = sprintf('train_%dQAM.mat', M);
    if ~isfile(dataFile)
        generateTrainingData(M, 0:2:14, 1000, dataFile);
    end
    mlpNet = trainMLP(dataFile, 2, 10, 300);
    gruNet = trainGRU(dataFile, mlpNet, 2, 100, 40);

    % --- BER curves ---
    ber_convLAS = simulateBER('convlas-hard', M, SNRdB_range, nBlocks);
    ber_mlpOnly = simulateBER('mlp-only',     M, SNRdB_range, nBlocks, mlpNet);
    ber_softLAS = simulateBER('softlas',      M, SNRdB_range, nBlocks);
    ber_deepLAS = simulateBER('deeplas',      M, SNRdB_range, nBlocks, mlpNet, gruNet);

    tag = sprintf('%dQAM', M);
    semilogy(SNRdB_range, ber_convLAS, '-o', 'DisplayName', ['Conv. LAS ' tag]);
    semilogy(SNRdB_range, ber_mlpOnly, '-s', 'DisplayName', ['MLP only ' tag]);
    semilogy(SNRdB_range, ber_softLAS, '-^', 'DisplayName', ['Prop. Soft-LAS ' tag]);
    semilogy(SNRdB_range, ber_deepLAS, '-d', 'DisplayName', ['Prop. Deep LAS ' tag]);
end

xlabel('SNR [dB]'); ylabel('Bit Error Rate');
legend('Location','southwest'); ylim([1e-5 1]);
title('Fig. 9 style: Deep LAS vs two-step Soft-LAS vs Conv. LAS');
