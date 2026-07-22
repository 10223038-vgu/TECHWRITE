%% run_fig11_sweeps.m
% Reproduces the structure of Fig. 11: (a) BER vs SNR for different
% FFT lengths, (b) BER vs SNR for different numbers of Tx/Rx antennas.
%
% NOTE: the model-based two-step Soft-LAS is agnostic to FFT length
% (FFT length only affects how many OFDM symbols are pooled into one
% training sequence for the GRU block per the paper's text -- for
% Fig. 11a you are really sweeping the *Deep LAS training* sequence
% length, not the per-symbol detector). This script demonstrates the
% antenna-count sweep (Fig. 11b) directly since it maps cleanly onto
% simulateBER.m; for Fig. 11a, regenerate training data with longer
% pooled sequences per FFT_len and retrain trainGRU.m accordingly.

clear; clc;
SNRdB_range = 0:1:10;   % finer step (was 0:2:10) -- smoother curve
nBlocks = 800;          % more samples per SNR point (was 300)
nRuns = 3;              % repeat and average across independent runs
M = 4;

antennaCounts = [4 8 16 32];   % paper sweeps up to 256; start smaller,
                                % LAS complexity is O(Nt^2) per symbol
figure; hold on; set(gca,'YScale','log'); grid on;

for Nt_test = antennaCounts
    ber_runs = zeros(nRuns, numel(SNRdB_range));
    qh = qamHelpers(); Es = qh.symEnergy(M);

    for r = 1:nRuns
        for si = 1:numel(SNRdB_range)
            snrLin = 10^(SNRdB_range(si)/10);
            sigma2 = Es/snrLin;
            nErr = 0; nBits = 0;
            for b = 1:nBlocks
                symIdx = randi([0 M-1], Nt_test, 1);
                x = qh.mod(symIdx, M);
                H = genChannel(Nt_test, Nt_test);
                n = sqrt(sigma2/2)*(randn(Nt_test,1)+1i*randn(Nt_test,1));
                y = H*x + n;
                [~, LLR] = softOutputLAS(y, H, sigma2, M, 'mmse', 50);
                trueBits = symbolsToBits(x, M);
                hardBits = double(LLR < 0);
                nErr = nErr + sum(sum(hardBits ~= trueBits));
                nBits = nBits + numel(trueBits);
            end
            ber_runs(r, si) = nErr/nBits;
        end
        fprintf('Nt=Nr=%d, run %d/%d done.\n', Nt_test, r, nRuns);
    end

    ber = mean(ber_runs, 1);   % average across independent runs
    semilogy(SNRdB_range, ber, '-o', 'DisplayName', sprintf('Nt=Nr=%d', Nt_test));
end

xlabel('SNR [dB]'); ylabel('Bit Error Rate'); legend show;
title('Fig. 11b style: BER vs number of transmit-receive antennas');
