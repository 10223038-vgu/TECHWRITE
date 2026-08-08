function poolData = buildPooledContextFeatures(M, SNRdB_list, Lseq, N, L, tau, nRealizations, mlpNet, stride)
% BUILDPOOLEDCONTEXTFEATURES Build a fixed-size, PERMUTATION-INVARIANT
% feature set from the same correlated-subcarrier context the GRU uses,
% instead of an ordered sequence.
%
% *** WHY THIS EXISTS ***
% run_direction1_ablation.m's Part A (H2 ablation) showed:
%   - correlated vs. shuffled GRU: BER gap flips sign across the SNR
%     sweep, tiny in magnitude -- statistically indistinguishable. The
%     GRU is NOT using context order/temporal structure.
%   - correlated (and shuffled) vs. i.i.d.: BER cut by >2x at every SNR,
%     consistently -- having CORRELATED context (regardless of order)
%     is genuinely useful.
% Conclusion: the useful signal is "these Lseq-1 neighboring subcarriers
% are correlated with the target," not "in this specific order." A GRU
% is the wrong tool for an order-invariant function -- it has to spend
% its limited training data learning to approximate permutation-
% invariance via recurrence, instead of getting it for free. This
% builder instead computes explicit order-invariant summary statistics
% (mean/max/std) over the context, which:
%   - is mathematically guaranteed permutation-invariant (unlike a GRU,
%     which merely happened to learn something close to invariant here)
%   - trains as a plain feedforward network (no BPTT, no vanishing/
%     exploding gradients over 64 steps, far more sample-efficient)
%   - should match or beat the GRU if H2's weaker form (correlation
%     content matters, order doesn't) is the real explanation
%
% Uses the exact same channel generation / SNR-range / sliding-window
% approach as buildCorrelatedAndShuffledSequences.m so results are
% directly comparable -- only the final feature representation differs.
%
% Inputs: same as buildCorrelatedAndShuffledSequences.m.
%
% Output: poolData struct with fields
%   .X - (numSamples x 4*(2Nt+1)) matrix, rows = observations, columns =
%        [target_UG ; mean(context) ; max(context) ; std(context)],
%        each a (2Nt+1)-length block. Row layout matches what
%        featureInputLayer + trainNetwork expects (numObservations x
%        numFeatures).
%   .Y - (numSamples x Nt*log2(M)) matrix of target LLR vectors, same
%        convention as buildCorrelatedAndShuffledSequences.m's .Y

if nargin < 9 || isempty(stride), stride = Lseq; end

cfg = getConfig();
Nt = cfg.Nt; Nr = cfg.Nr;
qh = qamHelpers();
Es = qh.symEnergy(M);
B = log2(M);
D = 2*Nt + 1;

if N < Lseq
    error('buildPooledContextFeatures:badN', 'N must be >= Lseq.');
end

startIdxList = 1:stride:(N - Lseq + 1);
nSeqPerRealization = numel(startIdxList);
totalSeq = nRealizations * nSeqPerRealization;

X = zeros(totalSeq, 4*D);
Y = zeros(totalSeq, Nt*B);

nSNR = numel(SNRdB_list);
seqCount = 0;
for r = 1:nRealizations
    snrdB = SNRdB_list(mod(r-1, nSNR) + 1);
    sigma2 = Es / 10^(snrdB/10);

    Hk_all = genMultipathChannel(Nr, Nt, N, L, tau);

    UG_full = zeros(D, N);
    LLR_full = zeros(Nt*B, N);
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

        [~, LLRtrue] = softOutputLAS(y, H, sigma2, M, 'mmse', cfg.maxLASIter);
        LLR_full(:, k) = reshape(LLRtrue.', [], 1);
    end

    for si = 1:nSeqPerRealization
        seqCount = seqCount + 1;
        idxRange = startIdxList(si) : startIdxList(si) + Lseq - 1;
        targetIdx = idxRange(end);
        contextIdx = idxRange(1:end-1);

        targetFeat = UG_full(:, targetIdx);              % (D x 1)
        contextFeat = UG_full(:, contextIdx);             % (D x Lseq-1)

        pooled = [targetFeat; ...
                  mean(contextFeat, 2); ...
                  max(contextFeat, [], 2); ...
                  std(contextFeat, 0, 2)];                 % (4D x 1)

        X(seqCount, :) = pooled.';
        Y(seqCount, :) = LLR_full(:, targetIdx).';
    end
end

poolData.X = X;
poolData.Y = Y;
end
