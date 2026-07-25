function [seqCorrelated, seqShuffled] = buildCorrelatedAndShuffledSequences(M, snrdB, Lseq, N, L, tau, nRealizations, mlpNet)
% BUILDCORRELATEDANDSHUFFLEDSEQUENCES Build the paired dataset needed for
% THEORY.md Section 1.7's critical ablation: two GRU training sets built
% from EXACTLY THE SAME underlying samples, differing only in whether the
% context timesteps are presented in true subcarrier (correlation) order
% or randomly shuffled.
%
% DESIGN (important for the ablation to be valid): for every sequence of
% length Lseq, the TARGET (the sample whose LLR is being predicted) is
% always the last timestep and is IDENTICAL in both arms -- only the
% preceding Lseq-1 "context" timesteps are reordered. This means both
% arms are asked to solve the exact same prediction problem (same target
% sample, same set of context evidence); the only variable under test is
% whether the context is presented in its true correlation order or not.
% If correlated beats shuffled despite identical content, that isolates
% the mechanism (the GRU is using the order/correlation), not just
% benefiting from a longer receptive field or more parameters.
%
% Requires original/deep-las-matlab on the MATLAB path (uses qamHelpers,
% getConfig, initEstimateSoft, softOutputLAS from there).
%
% Inputs:
%   M             - QAM order
%   snrdB         - SNR (dB) for this dataset
%   Lseq          - sequence length (set via coherenceBandwidth.m)
%   N             - number of subcarriers per channel realization (must be
%                   >= Lseq; N should be >> Lseq if you want each
%                   realization to contribute multiple non-overlapping
%                   sequences)
%   L, tau        - multipath channel parameters (genMultipathChannel.m)
%   nRealizations - number of independent channel realizations to draw
%   mlpNet        - trained MLP block (from original/deep-las-matlab's
%                   trainMLP.m on the ORIGINAL flat i.i.d. dataset -- this
%                   is valid to reuse as-is without retraining because
%                   each H_k is still marginally CN(0,1), see
%                   genMultipathChannel.m's header comment)
%
% Outputs: seqCorrelated, seqShuffled -- each a struct with fields
%   .X - cell array of (2Nt+1) x Lseq feature sequences
%   .Y - matrix of Nt*log2(M) x 1 target LLR vectors (N_seq x NtB)

cfg = getConfig();
Nt = cfg.Nt; Nr = cfg.Nr;
qh = qamHelpers();
Es = qh.symEnergy(M);
sigma2 = Es / 10^(snrdB/10);
B = log2(M);

if N < Lseq
    error('buildCorrelatedAndShuffledSequences:badN', 'N must be >= Lseq.');
end

nSeqPerRealization = floor(N / Lseq);
totalSeq = nRealizations * nSeqPerRealization;

XCorr = cell(totalSeq, 1); YCorr = zeros(totalSeq, Nt*B);
XShuf = cell(totalSeq, 1); YShuf = zeros(totalSeq, Nt*B);

seqCount = 0;
for r = 1:nRealizations
    Hk_all = genMultipathChannel(Nr, Nt, N, L, tau);

    UG_full = zeros(2*Nt+1, N);
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

    for s = 1:nSeqPerRealization
        seqCount = seqCount + 1;
        idxRange = (s-1)*Lseq + 1 : s*Lseq;
        targetIdx = idxRange(end);          % fixed target, identical in both arms
        contextIdx = idxRange(1:end-1);      % the part that gets reordered

        % --- correlated arm: context in true consecutive order ---
        seqIdxCorr = [contextIdx, targetIdx];
        XCorr{seqCount} = UG_full(:, seqIdxCorr);
        YCorr(seqCount, :) = LLR_full(:, targetIdx).';

        % --- shuffled arm: SAME context samples, random order; SAME target ---
        shuffledContext = contextIdx(randperm(numel(contextIdx)));
        seqIdxShuf = [shuffledContext, targetIdx];
        XShuf{seqCount} = UG_full(:, seqIdxShuf);
        YShuf(seqCount, :) = LLR_full(:, targetIdx).';  % identical target/content to correlated arm
    end
end

seqCorrelated.X = XCorr; seqCorrelated.Y = YCorr;
seqShuffled.X = XShuf; seqShuffled.Y = YShuf;
end
