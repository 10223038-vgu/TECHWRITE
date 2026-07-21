function BER = simulateBER(detectorType, M, SNRdB_range, nBlocks, mlpNet, gruNet)
% SIMULATEBER Monte-Carlo (uncoded) bit-error-rate simulation for a
% chosen detector. Use this to reproduce the *detector comparison*
% curves of Figs. 9-12 (relative SNR gaps between detectors).
%
% NOTE ON CHANNEL CODING: the paper's BER curves are measured after
% turbo decoding (rate 1/2, 8 iterations) with a random interleaver.
% This function reports UNCODED bit errors at the detector output
% (i.e., BER of the raw LLR/hard decisions), which preserves the
% *relative* SNR gaps between detectors (the whole point of Figs.
% 9-12) without depending on an exact, unspecified turbo puncturing
% pattern. To reproduce absolute BER numbers with channel coding,
% wrap the LLR output of this function's detector calls with
% comm.TurboEncoder / comm.TurboDecoder (see README.md, "Adding
% Turbo Coding").
%
% detectorType: 'convlas-hard' | 'softlas' | 'deeplas' | 'mlp-only' | 'mmse-hard'
% mlpNet, gruNet: required only for 'deeplas' / 'mlp-only'

cfg = getConfig();
Nt = cfg.Nt; Nr = cfg.Nr;
qh = qamHelpers();
Es = qh.symEnergy(M);

BER = zeros(size(SNRdB_range));

for si = 1:numel(SNRdB_range)
    snrdB = SNRdB_range(si);
    snrLin = 10^(snrdB/10);
    sigma2 = Es / snrLin;

    nErr = 0; nBits = 0;

    for b = 1:nBlocks
        symIdx = randi([0 M-1], Nt, 1);
        x = qh.mod(symIdx, M);
        H = genChannel(Nr, Nt);
        n = sqrt(sigma2/2) * (randn(Nr,1) + 1i*randn(Nr,1));
        y = H*x + n;

        trueBits = symbolsToBits(x, M);

        switch lower(detectorType)
            case 'mmse-hard'
                xhat = initEstimate(y, H, sigma2, M, 'mmse');
                hardBits = symbolsToBits(xhat, M);

            case 'convlas-hard'
                [Hr, yr] = complexToReal(H, y);
                xhat0 = initEstimate(y, H, sigma2, M, 'mmse');
                xr0 = [real(xhat0); imag(xhat0)];
                xr_hat = las1Hard(yr, Hr, xr0, M, cfg.maxLASIter);
                xhat = xr_hat(1:Nt) + 1i*xr_hat(Nt+1:end);
                hardBits = symbolsToBits(xhat, M);

            case 'softlas'
                [~, LLR] = softOutputLAS(y, H, sigma2, M, 'mmse', cfg.maxLASIter);
                hardBits = double(LLR < 0);

            case 'deeplas'
                LLR = deepLASPredict(y, H, sigma2, M, mlpNet, gruNet);
                hardBits = double(LLR < 0);

            case 'mlp-only'
                xhat_mmse = initEstimate(y, H, sigma2, M, 'mmse');
                xn = [real(xhat_mmse); imag(xhat_mmse)];
                xn = xn ./ max(abs(xn));
                llrVec = mlpNet(xn);
                LLR = reshape(llrVec, log2(M), Nt).';
                hardBits = double(LLR < 0);

            otherwise
                error('simulateBER:badDetector', 'Unknown detectorType: %s', detectorType);
        end

        nErr = nErr + sum(sum(hardBits ~= trueBits));
        nBits = nBits + numel(trueBits);
    end

    BER(si) = nErr / nBits;
    fprintf('[%s, %d-QAM] SNR=%2d dB -> BER=%.3e\n', detectorType, M, snrdB, BER(si));
end
end
