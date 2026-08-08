function [seqCorrelated, seqShuffled] = buildCorrelatedAndShuffledSequences(M, SNRdB_list, Lseq, N, L, tau, nRealizations, mlpNet, stride)
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
%   SNRdB_list    - SNR (dB) VALUE(S) for this dataset. Pass a VECTOR
%                   spanning (at least) the SNR range you will later
%                   evaluate BER over -- e.g. the same SNRdB_range used
%                   in simulateBERDirection1.m -- not a single scalar.
%                   Realizations cycle through this list round-robin, so
%                   the pooled training set spans the same noise range
%                   the original codebase's generateTrainingData.m uses
%                   (0:2:14 dB). Training at a single fixed SNR (the old
%                   default) and then testing across a wide SNR sweep is
%                   an out-of-distribution mismatch: the GRU never learns
%                   what very-low-noise or very-high-noise inputs look
%                   like, so it degrades toward guessing away from that
%                   one training point -- this is what "still doing
%                   random guessing" after the sequence-length fix looks
%                   like. A scalar is still accepted (single-SNR dataset)
%                   for backward compatibility / deliberate SNR-specific
%                   experiments, but is no longer the recommended default.
%   Lseq          - sequence length (set via coherenceBandwidth.m)
%   N             - number of subcarriers per channel realization (must be
%                   >= Lseq)
%   L, tau        - multipath channel parameters (genMultipathChannel.m)
%   nRealizations - number of independent channel realizations to draw
%                   (TOTAL across all SNRs in SNRdB_list, cycled
%                   round-robin -- not per SNR point)
%   mlpNet        - trained MLP block (from original/deep-las-matlab's
%                   trainMLP.m on the ORIGINAL flat i.i.d. dataset -- this
%                   is valid to reuse as-is without retraining because
%                   each H_k is still marginally CN(0,1), see
%                   genMultipathChannel.m's header comment)
%   stride        - step (in subcarriers) between the START of consecutive
%                   training windows within one realization. DEFAULT =
%                   Lseq (the original non-overlapping behavior: only
%                   floor(N/Lseq) windows per realization -- e.g. just 4
%                   windows per realization at N=256, Lseq=64, which with
%                   a few hundred realizations gives only a few hundred
%                   training sequences total for a 2-layer/100-unit GRU --
%                   badly data-starved). Pass a SMALLER stride (e.g. 1-8)
%                   to slide overlapping windows across each realization
%                   instead: this reuses the SAME already-computed
%                   per-subcarrier channel/LLR data (the expensive part --
%                   one softOutputLAS call per subcarrier) to produce up
%                   to (N-Lseq)/stride+1 windows per realization for
%                   ~free, multiplying the effective training-set size
%                   without a proportional increase in simulation cost
%                   (only the downstream GRU training time scales with
%                   the larger dataset).
%
% Outputs: seqCorrelated, seqShuffled -- each a struct with fields
%   .X - cell array of (2Nt+1) x Lseq feature sequences
%   .Y - matrix of Nt*log2(M) x 1 target LLR vectors (N_seq x NtB)

if nargin < 9 || isempty(stride), stride = Lseq; end

cfg = getConfig();
Nt = cfg.Nt; Nr = cfg.Nr;
qh = qamHelpers();
Es = qh.symEnergy(M);
B = log2(M);

if N < Lseq
    error('buildCorrelatedAndShuffledSequences:badN', 'N must be >= Lseq.');
end

startIdxList = 1:stride:(N - Lseq + 1);
nSeqPerRealization = numel(startIdxList);
totalSeq = nRealizations * nSeqPerRealization;

XCorr = cell(totalSeq, 1); YCorr = zeros(totalSeq, Nt*B);
XShuf = cell(totalSeq, 1); YShuf = zeros(totalSeq, Nt*B);

nSNR = numel(SNRdB_list);
seqCount = 0;
for r = 1:nRealizations
    snrdB = SNRdB_list(mod(r-1, nSNR) + 1);   % round-robin through the SNR list
    sigma2 = Es / 10^(snrdB/10);

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

    for si = 1:nSeqPerRealization
        seqCount = seqCount + 1;
        idxRange = startIdxList(si) : startIdxList(si) + Lseq - 1;
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
