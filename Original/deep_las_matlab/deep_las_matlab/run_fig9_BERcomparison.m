%% run_fig9_BERcomparison.m
% Reproduces the structure of Fig. 9: BER vs SNR for the conventional
% hard-output 1-LAS, the proposed model-based two-step soft-output
% LAS, the proposed data-driven Deep LAS, and the standalone MLP.
%
% Run generate + train scripts first (see README.md, Quick Start).

clear; clc;
cfg = getConfig();
SNRdB_range = 0:2:14;
nBlocks = 2000;    % raise further for smoother, lower-BER curves

figure; hold on; set(gca,'YScale','log'); grid on;
colors = lines(8);

for M = cfg.M_list
    fprintf('\n=== M = %d-QAM ===\n', M);

    % --- Train Deep LAS for this modulation order (or load if cached) ---
    dataFile = sprintf('train_%dQAM.mat', M);
    if ~isfile(dataFile)
        generateTrainingData(M, 0:2:14, 3000, dataFile);   % more samples/SNR
    end
    mlpNet = trainMLP(dataFile, 2, 10, 300);
    gruNet = trainGRU(dataFile, mlpNet, 2, 100, 40);

    % --- BER curves ---
    % 'mmse-hard' is a plain linear-detector sanity baseline (no LAS,
    % no ML) -- if MLP-only / Deep LAS ever sit ABOVE this line, that's
    % a red flag the network has learned nothing useful, since even a
    % trivial linear detector should beat "no detector."
    ber_mmse    = simulateBER('mmse-hard',    M, SNRdB_range, nBlocks);
    ber_convLAS = simulateBER('convlas-hard', M, SNRdB_range, nBlocks);
    ber_mlpOnly = simulateBER('mlp-only',     M, SNRdB_range, nBlocks, mlpNet);
    ber_softLAS = simulateBER('softlas',      M, SNRdB_range, nBlocks);
    ber_deepLAS = simulateBER('deeplas',      M, SNRdB_range, nBlocks, mlpNet, gruNet);

    tag = sprintf('%dQAM', M);
    semilogy(SNRdB_range, ber_mmse,    ':x', 'DisplayName', ['MMSE hard (sanity) ' tag]);
    semilogy(SNRdB_range, ber_convLAS, '-o', 'DisplayName', ['Conv. LAS ' tag]);
    semilogy(SNRdB_range, ber_mlpOnly, '-s', 'DisplayName', ['MLP only ' tag]);
    semilogy(SNRdB_range, ber_softLAS, '-^', 'DisplayName', ['Prop. Soft-LAS ' tag]);
    semilogy(SNRdB_range, ber_deepLAS, '-d', 'DisplayName', ['Prop. Deep LAS ' tag]);
end

xlabel('SNR [dB]'); ylabel('Bit Error Rate');
legend('Location','southwest'); ylim([1e-5 1]);
title('Fig. 9 style: Deep LAS vs two-step Soft-LAS vs Conv. LAS');
