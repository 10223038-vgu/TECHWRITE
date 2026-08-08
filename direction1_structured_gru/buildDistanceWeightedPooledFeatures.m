function poolData = buildDistanceWeightedPooledFeatures(M, SNRdB_list, Lseq, N, L, tau, nRealizations, mlpNet, stride, nearFrac, decayScale)
% BUILDDISTANCEWEIGHTEDPOOLEDFEATURES Distance-aware extension of
% buildPooledContextFeatures.m, per THEORY_POOLING.md Section 6.
%
% *** WHY THIS EXISTS ***
% buildPooledContextFeatures.m's mean/max/std pooling treats every one of
% the Lseq-1 context subcarriers identically, regardless of how far it is
% from the target. But THEORY.md Sections 1.4/1.6 are built entirely on
% the premise that correlation strength DECAYS with distance |delta| from
% the target subcarrier (that's the whole reason Lseq was set to the
% coherence bandwidth Bc in the first place). Pooling uniformly across
% the window blends a strongly-correlated near neighbor with a nearly-
% decorrelated far one at equal weight, diluting exactly the signal this
% whole direction is trying to exploit. This builder reintroduces that
% distance structure via THREE mechanisms, still combined only with
% permutation-invariant (order-independent WITHIN each group) operations,
% consistent with the H2 ablation's finding that order doesn't matter:
%
%   1. NEAR/FAR SPLIT POOLING: mean/max/std computed separately over a
%      "near" window (the nearFrac closest context subcarriers to the
%      target) and the remaining "far" context, instead of one pool over
%      everything. Gives the network two differently-scoped aggregates.
%   2. DISTANCE-WEIGHTED MEAN: an additional exponentially-decaying-
%      weighted mean, weight(delta) = exp(-delta/decayScale), so even
%      within a single aggregate, closer context contributes more.
%   3. The target's own feature vector is still always passed through
%      unpooled (as in buildPooledContextFeatures.m).
%
% Inputs: same as buildPooledContextFeatures.m, plus:
%   nearFrac    - fraction (0,1) of the Lseq-1 context, BY DISTANCE, that
%                 counts as "near" (the nearFrac*Lseq closest-to-target
%                 subcarriers). Default 0.25 (quarter window).
%   decayScale  - decay constant for the distance-weighted mean, in units
%                 of subcarrier spacing. Default Lseq/2: at the window's
%                 farthest distance (~Lseq-1), weight ~ exp(-2) ~= 0.135
%                 -- still non-negligible (this window is, by
%                 construction, the coherence bandwidth, so correlation
%                 shouldn't collapse to ~0 at the far edge) but clearly
%                 down-weighted relative to near context.
%
% Output: poolData struct with fields .X, .Y (same convention as
% buildPooledContextFeatures.m). Feature layout of .X's columns:
%   [target ; nearMean ; nearMax ; nearStd ; farMean ; farMax ; farStd ;
%    distWeightedMean]   -- 8 blocks of (2Nt+1) each.

if nargin < 9  || isempty(stride),     stride = Lseq; end
if nargin < 10 || isempty(nearFrac),   nearFrac = 0.25; end
if nargin < 11 || isempty(decayScale), decayScale = Lseq/2; end

cfg = getConfig();
Nt = cfg.Nt; Nr = cfg.Nr;
qh = qamHelpers();
Es = qh.symEnergy(M);
B = log2(M);
D = 2*Nt + 1;

if N < Lseq
    error('buildDistanceWeightedPooledFeatures:badN', 'N must be >= Lseq.');
end

startIdxList = 1:stride:(N - Lseq + 1);
nSeqPerRealization = numel(startIdxList);
totalSeq = nRealizations * nSeqPerRealization;

X = zeros(totalSeq, 8*D);
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
        contextIdx = idxRange(1:end-1);           % Lseq-1 preceding subcarriers
        distToTarget = targetIdx - contextIdx;      % all positive, 1..Lseq-1, INCREASING with position

        pooled = pooledFeaturesForOneWindow(UG_full, targetIdx, contextIdx, distToTarget, nearFrac, decayScale);
        X(seqCount, :) = pooled.';
        Y(seqCount, :) = LLR_full(:, targetIdx).';
    end
end

poolData.X = X;
poolData.Y = Y;
end
