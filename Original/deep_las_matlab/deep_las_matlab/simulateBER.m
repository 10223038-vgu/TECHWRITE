function BER = simulateBER(detectorType, M, SNRdB_range, nBlocksMin, mlpNet, gruNet, minErrors, maxBlocksFactor, initType)
% SIMULATEBER Monte-Carlo (uncoded) bit-error-rate simulation for a
% chosen detector. Use this to reproduce the *detector comparison*
% curves of Figs. 9-12 (relative SNR gaps between detectors).
%
% ADAPTIVE STOPPING (fixes shallow/noisy high-SNR tail): a FIXED block
% count sees very few actual bit errors at high SNR (e.g. at 12-14 dB
% you may see 0-2 errors out of a few thousand bits), which makes the
% BER estimate at those points extremely noisy -- this is very likely
% why your Fig. 9 curves weren't dropping as cleanly as the paper's.
% Standard practice is to keep simulating until you've observed a
% minimum number of actual errors (target ~100-200) at every SNR
% point, capped by a maximum block count so low-BER points don't run
% forever. This function now does that: it always runs at least
% nBlocksMin blocks, then keeps going until either minErrors errors
% have been observed or maxBlocksFactor*nBlocksMin blocks have run.
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
% detectorType: 'convlas-hard' | 'softlas' | 'deeplas' | 'mlp-only' | 'mmse-hard' | 'sd-optimal'
% mlpNet, gruNet: required only for 'deeplas' / 'mlp-only'
% minErrors: target minimum bit errors per SNR point (default 150)
% maxBlocksFactor: hard cap = maxBlocksFactor * nBlocksMin blocks (default 40)

if nargin < 7 || isempty(minErrors),       minErrors = 150; end
if nargin < 8 || isempty(maxBlocksFactor), maxBlocksFactor = 40; end
if nargin < 9 || isempty(initType),        initType = 'mmse'; end
if nargin < 5, mlpNet = []; end
if nargin < 6, gruNet = []; end

cfg = getConfig();
Nt = cfg.Nt; Nr = cfg.Nr;
qh = qamHelpers();
Es = qh.symEnergy(M);

BER = zeros(size(SNRdB_range));
maxBlocks = maxBlocksFactor * nBlocksMin;

for si = 1:numel(SNRdB_range)
    snrdB = SNRdB_range(si);
    snrLin = 10^(snrdB/10);
    sigma2 = Es / snrLin;

    nErr = 0; nBits = 0; nBlocksRun = 0;

    while true
        batchSize = nBlocksMin;   % run in batches, re-check stopping criteria
        for b = 1:batchSize
            symIdx = randi([0 M-1], Nt, 1);
            x = qh.mod(symIdx, M);
            H = genChannel(Nr, Nt);
            n = sqrt(sigma2/2) * (randn(Nr,1) + 1i*randn(Nr,1));
            y = H*x + n;

            trueBits = symbolsToBits(x, M);
            hardBits = detectHardBits(detectorType, y, H, sigma2, M, Nt, cfg, mlpNet, gruNet, initType);

            nErr = nErr + sum(sum(hardBits ~= trueBits));
            nBits = nBits + numel(trueBits);
        end
        nBlocksRun = nBlocksRun + batchSize;

        if nErr >= minErrors || nBlocksRun >= maxBlocks
            break;
        end
    end

    BER(si) = nErr / nBits;
    fprintf('[%s, %d-QAM] SNR=%2d dB -> BER=%.3e (%d errors / %d bits, %d blocks)\n', ...
        detectorType, M, snrdB, BER(si), nErr, nBits, nBlocksRun);
end
end

% ------------------------------------------------------------------
function hardBits = detectHardBits(detectorType, y, H, sigma2, M, Nt, cfg, mlpNet, gruNet, initType)
switch lower(detectorType)
    case 'mmse-hard'
        xhat = initEstimate(y, H, sigma2, M, initType);
        hardBits = symbolsToBits(xhat, M);

    case 'convlas-hard'
        [Hr, yr] = complexToReal(H, y);
        xhat0 = initEstimate(y, H, sigma2, M, initType);
        xr0 = [real(xhat0); imag(xhat0)];
        xr_hat = las1Hard(yr, Hr, xr0, M, cfg.maxLASIter);
        xhat = xr_hat(1:Nt) + 1i*xr_hat(Nt+1:end);
        hardBits = symbolsToBits(xhat, M);

    case 'softlas'
        [~, LLR] = softOutputLAS(y, H, sigma2, M, initType, cfg.maxLASIter);
        hardBits = double(LLR < 0);

    case 'deeplas'
        LLR = deepLASPredict(y, H, sigma2, M, mlpNet, gruNet);
        hardBits = double(LLR < 0);

    case 'mlp-only'
        xhat_mmse_soft = initEstimateSoft(y, H, sigma2, M, 'mmse');
        xn = [real(xhat_mmse_soft); imag(xhat_mmse_soft)];
        xn = xn ./ max(abs(xn));
        llrVec = mlpNet(xn);
        LLR = reshape(llrVec, log2(M), Nt).';
        hardBits = double(LLR < 0);

    case 'sd-optimal'
        [~, LLR] = bruteForceSD(y, H, sigma2, M);
        hardBits = double(LLR < 0);

    case 'detnet'
        % gruNet argument slot is reused to pass the trained DetNet
        % struct (from trainDetNet.m). DetNet is a hard-output
        % detector, so this returns bits directly, not an LLR.
        hardBits = detNetPredict(y, H, gruNet);

    otherwise
        error('simulateBER:badDetector', 'Unknown detectorType: %s', detectorType);
end
end
