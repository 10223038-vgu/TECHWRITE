function BER = simulateBERDistancePooling(M, SNRdB_range, nBlocksMin, mlpNet, poolNet, Lseq, N, L, tau, nearFrac, decayScale, minErrors, maxBlocksFactor)
% SIMULATEBERDISTANCEPOOLING BER simulator for the distance-aware
% pooling-net Deep LAS detector. Same role as simulateBERPooling.m, but
% builds the distance-weighted/near-far-split pooled features (see
% buildDistanceWeightedPooledFeatures.m) via the SAME shared
% pooledFeaturesForOneWindow.m helper used at training time, so
% train/test feature construction cannot drift apart.

if nargin < 10 || isempty(nearFrac),        nearFrac = 0.25;   end
if nargin < 11 || isempty(decayScale),       decayScale = Lseq/2; end
if nargin < 12 || isempty(minErrors),       minErrors = 150; end
if nargin < 13 || isempty(maxBlocksFactor), maxBlocksFactor = 40; end

cfg = getConfig();
Nt = cfg.Nt; Nr = cfg.Nr;
qh = qamHelpers();
Es = qh.symEnergy(M);
B = log2(M);
D = 2*Nt + 1;

if N < Lseq
    error('simulateBERDistancePooling:badN', 'N must be >= Lseq.');
end
nSeqPerRealization = floor(N / Lseq);
if nSeqPerRealization < 1
    error('simulateBERDistancePooling:badLseq', 'Lseq must be <= N.');
end

BER = zeros(size(SNRdB_range));
maxBlocks = maxBlocksFactor * nBlocksMin;

for si = 1:numel(SNRdB_range)
    snrdB = SNRdB_range(si);
    sigma2 = Es / 10^(snrdB/10);

    nErr = 0; nBits = 0; nTargetsRun = 0;

    while true
        Hk_all = genMultipathChannel(Nr, Nt, N, L, tau);

        UG_full = zeros(D, N);
        trueBits_full = zeros(Nt*B, N);
        for k = 1:N
            H = squeeze(Hk_all(k, :, :));
            symIdx = randi([0 M-1], Nt, 1);
            x = qh.mod(symIdx, M);
            n = sqrt(sigma2/2) * (randn(Nr,1) + 1i*randn(Nr,1));
            y = H*x + n;

            xhatSoft = initEstimateSoft(y, H, sigma2, M, 'mmse');
            xn = [real(xhatSoft); imag(xhatSoft)];
            xn = xn ./ max(abs(xn));
            llrRough = mean(mlpNet(xn));
            UG_full(:, k) = [xn; llrRough];

            trueBits_full(:, k) = reshape(symbolsToBits(x, M).', [], 1);
        end

        for s = 1:nSeqPerRealization
            idxRange = (s-1)*Lseq + 1 : s*Lseq;
            targetIdx = idxRange(end);
            contextIdx = idxRange(1:end-1);
            distToTarget = targetIdx - contextIdx;

            pooled = pooledFeaturesForOneWindow(UG_full, targetIdx, contextIdx, distToTarget, nearFrac, decayScale);

            llrRow = predict(poolNet, pooled.');   % 1 x (Nt*B)
            llrVec = llrRow(:);
            LLR = reshape(llrVec, B, Nt).';
            hardBits = double(LLR < 0);

            trueBitsTarget = reshape(trueBits_full(:, targetIdx), B, Nt).';

            nErr = nErr + sum(sum(hardBits ~= trueBitsTarget));
            nBits = nBits + numel(trueBitsTarget);
            nTargetsRun = nTargetsRun + 1;
        end

        if nErr >= minErrors || nTargetsRun >= maxBlocks
            break;
        end
    end

    BER(si) = nErr / nBits;
    fprintf('[distance-pooling-net, %d-QAM] SNR=%2d dB -> BER=%.3e (%d errors / %d bits, %d target subcarriers)\n', ...
        M, snrdB, BER(si), nErr, nBits, nTargetsRun);
end
end
