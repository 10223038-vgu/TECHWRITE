%% run_fig11_sweeps.m
% Reproduces Fig. 11: (a) BER vs SNR for different FFT lengths,
% (b) BER vs SNR for different numbers of Tx/Rx antennas.
%
% FIG. 11(a) IMPLEMENTATION NOTE: the paper explains that longer FFT
% lengths give the GRU block more pooled input per training sequence,
% improving its LLR approximation (Section V, discussion of Fig. 11).
% This is now implemented literally: trainGRU.m accepts a `seqLen`
% argument that pools `seqLen` consecutive dataset samples into one
% multi-timestep training sequence. Here FFT_len -> seqLen via the
% proxy seqLen = round(FFT_len/256) (512->2, 1024->4, 2048->8) -- an
% interpretive choice (the paper doesn't give an exact formula), but
% it reproduces the qualitative story: longer FFT length -> longer
% pooled training sequences -> better-trained GRU -> lower BER.
%
% Both panels now use simulateBER.m's adaptive stopping (minimum error
% count per SNR point) and finer SNR steps, which should give visibly
% smoother/more continuous curves than the earlier fixed-block version.

clear; clc;
cfg = getConfig();
M = 4;
dataFile = sprintf('train_%dQAM.mat', M);
if ~isfile(dataFile)
    generateTrainingData(M, 0:2:14, 3000, dataFile);
end

SNRdB_range = 0:1:12;   % finer than before (was 0:2:10)
nBlocksMin = 500;       % adaptive stopping will run more as needed

figure('Name', 'Fig. 11: FFT length and antenna count sweeps');

%% (a) FFT length sweep (via GRU sequence-pooling proxy)
FFT_lengths = [512, 1024, 2048];
mlpNet = trainMLP(dataFile, 2, 10, 300);   % shared MLP block across the sweep

subplot(1,2,1); hold on; grid on; set(gca,'YScale','log');
for fftLen = FFT_lengths
    seqLen = max(1, round(fftLen/256));
    fprintf('\n--- FFT length = %d (seqLen=%d) ---\n', fftLen, seqLen);
    gruNet_fft = trainGRU(dataFile, mlpNet, 2, 100, 40, seqLen);
    ber = simulateBER('deeplas', M, SNRdB_range, nBlocksMin, mlpNet, gruNet_fft);
    semilogy(SNRdB_range, ber, '-o', 'DisplayName', sprintf('FFT length = %d', fftLen));
end
xlabel('SNR [dB]'); ylabel('Bit Error Rate'); legend show;
ylim([1e-5 1]);
title('Fig. 11(a): BER vs FFT length (4-QAM)');

%% (b) Antenna-count sweep
antennaCounts = [4 8 16 32];   % paper sweeps up to 256; start smaller,
                                % LAS complexity is O(Nt^2) per symbol
subplot(1,2,2); hold on; grid on; set(gca,'YScale','log');

for Nt_test = antennaCounts
    qh = qamHelpers(); Es = qh.symEnergy(M);
    ber = zeros(size(SNRdB_range));
    for si = 1:numel(SNRdB_range)
        snrLin = 10^(SNRdB_range(si)/10);
        sigma2 = Es/snrLin;
        nErr = 0; nBits = 0; nBlocksRun = 0;
        minErrors = 150; maxBlocks = 40*nBlocksMin;
        while true
            for b = 1:nBlocksMin
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
            nBlocksRun = nBlocksRun + nBlocksMin;
            if nErr >= minErrors || nBlocksRun >= maxBlocks
                break;
            end
        end
        ber(si) = nErr/nBits;
    end
    semilogy(SNRdB_range, ber, '-o', 'DisplayName', sprintf('Nt=Nr=%d', Nt_test));
    fprintf('Nt=Nr=%d sweep done.\n', Nt_test);
end

xlabel('SNR [dB]'); ylabel('Bit Error Rate'); legend show;
ylim([1e-5 1]);
title('Fig. 11(b): BER vs number of transmit-receive antennas');
