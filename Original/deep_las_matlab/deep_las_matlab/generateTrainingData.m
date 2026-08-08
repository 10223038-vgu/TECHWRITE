function generateTrainingData(M, SNRdB_list, nSamplesPerSNR, outFile)
% GENERATETRAININGDATA Build the {yhat, xhat_mmse} -> LLR dataset
% (Eq. 19) used to train the Deep LAS MLP+GRU network, with target
% LLR values produced by the model-based two-step soft-output LAS
% (softOutputLAS.m). Saves a .mat file with all samples pooled
% across the requested SNR points.
%
% Example:
%   generateTrainingData(4,  0:2:14, 3000, 'train_4QAM.mat');
%   generateTrainingData(16, 0:2:14, 3000, 'train_16QAM.mat');

cfg = getConfig();
Nt = cfg.Nt; Nr = cfg.Nr;
qh = qamHelpers();
Es = qh.symEnergy(M);

Yhat_all = [];   % (2*Nt) x nTotal   -- [Re(y);Im(y)]
Xm_all   = [];   % (2*Nt) x nTotal   -- [Re(xhat_mmse);Im(xhat_mmse)]
LLR_all  = [];   % (Nt*log2(M)) x nTotal

for snrdB = SNRdB_list
    snrLin = 10^(snrdB/10);
    sigma2 = Es / snrLin;         % per-real-dim noise variance convention

    for s = 1:nSamplesPerSNR
        bitsIdx = randi([0 M-1], Nt, 1);
        x = qh.mod(bitsIdx, M);
        H = genChannel(Nr, Nt);
        n = sqrt(sigma2/2) * (randn(Nr,1) + 1i*randn(Nr,1));
        y = H*x + n;

        xhat_mmse_soft = initEstimateSoft(y, H, sigma2, M, 'mmse');  % FIX: soft, not hard-decided
        [~, LLR_true] = softOutputLAS(y, H, sigma2, M, 'mmse', cfg.maxLASIter);

        yhat = [real(y); imag(y)];
        xm   = [real(xhat_mmse_soft); imag(xhat_mmse_soft)];
        llrVec = reshape(LLR_true.', [], 1);   % Nt*log2(M) x 1

        Yhat_all = [Yhat_all, yhat]; %#ok<AGROW>
        Xm_all   = [Xm_all,   xm];   %#ok<AGROW>
        LLR_all  = [LLR_all,  llrVec]; %#ok<AGROW>
    end
    fprintf('generateTrainingData: M=%d, SNR=%d dB done (%d samples)\n', ...
        M, snrdB, nSamplesPerSNR);
end

save(outFile, 'Yhat_all', 'Xm_all', 'LLR_all', 'M', 'SNRdB_list', 'Nt', 'Nr', '-v7.3');
fprintf('Saved dataset to %s (%d total samples)\n', outFile, size(Xm_all,2));
end
