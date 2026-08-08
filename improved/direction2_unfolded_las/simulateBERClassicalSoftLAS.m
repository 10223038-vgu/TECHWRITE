function BER = simulateBERClassicalSoftLAS(M, SNRdB_range, nBlocksMin, minErrors, maxBlocksFactor)
% SIMULATEBERCLASSICALSOFTLAS BER simulator for the CLASSICAL (fixed, no
% learning) softOutputLAS.m two-step detector -- Algorithm 1 + Algorithm
% 2, run to convergence, exactly as originally published. This is the
% direct baseline Unfolded-A/B should be compared against: since both
% unfolded variants START at alpha=1 for every layer (which reproduces
% classical per-layer behavior, see trainUnfoldedLAS.m), this comparison
% isolates "did learning the step sizes end-to-end against true bits
% help at all," independent of Direction 1's separate MLP-GRU question.

if nargin < 4 || isempty(minErrors),       minErrors = 150; end
if nargin < 5 || isempty(maxBlocksFactor), maxBlocksFactor = 40; end

cfg = getConfig();
Nt = cfg.Nt; Nr = cfg.Nr;
qh = qamHelpers();
Es = qh.symEnergy(M);

BER = zeros(size(SNRdB_range));
maxBlocks = maxBlocksFactor * nBlocksMin;

for si = 1:numel(SNRdB_range)
    snrdB = SNRdB_range(si);
    sigma2 = Es / 10^(snrdB/10);

    nErr = 0; nBits = 0; nBlocksRun = 0;

    while true
        H = genChannel(Nr, Nt);
        symIdx = randi([0 M-1], Nt, 1);
        x = qh.mod(symIdx, M);
        n = sqrt(sigma2/2) * (randn(Nr,1) + 1i*randn(Nr,1));
        y = H*x + n;

        [~, LLR] = softOutputLAS(y, H, sigma2, M, 'mmse', cfg.maxLASIter);
        trueBits = symbolsToBits(x, M);
        hardBits = double(LLR < 0);

        nErr = nErr + sum(sum(hardBits ~= trueBits));
        nBits = nBits + numel(trueBits);
        nBlocksRun = nBlocksRun + 1;

        if nErr >= minErrors || nBlocksRun >= maxBlocks
            break;
        end
    end

    BER(si) = nErr / nBits;
    fprintf('[classical-softLAS, %d-QAM] SNR=%2d dB -> BER=%.3e (%d errors / %d bits, %d blocks)\n', ...
        M, snrdB, BER(si), nErr, nBits, nBlocksRun);
end
end
