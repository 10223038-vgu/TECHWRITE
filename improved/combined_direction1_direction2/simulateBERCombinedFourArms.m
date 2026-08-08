function BER = simulateBERCombinedFourArms(M, SNRdB_range, nBlocksMin, mlpNet, gruNet_orig, poolNet, Lseq, N, L, tau, unfoldedParamsPlain, unfoldedParamsCombined, minErrors, maxBlocksFactor)
% SIMULATEBERCOMBINEDFOURARMS Evaluates all four arms of the required
% three-way (here: four-way, including the original paper's baseline)
% ablation on SHARED correlated-multipath channel realizations -- a
% paired comparison, same draws across arms, lower variance than
% evaluating each arm independently (matches how Direction 1's own H1/H2
% ablations were done).
%
% *** THE COMBINATION ***
% (a) Original Deep LAS       : deepLASPredict.m (unchanged, per-subcarrier, no context)
% (b) Direction-1-only        : frozen pooling net (mean/max/std over correlated context)
% (c) Direction-2-only        : Unfolded-B-Soft with unfoldedParamsPlain -- trained AND
%                                evaluated on the plain single-subcarrier hard MMSE
%                                estimate, exactly as validated standalone in Direction 2
% (d) COMBINED                : Unfolded-B-Soft with unfoldedParamsCombined -- trained
%                                AND evaluated on the pooling-informed initial estimate.
%                                Using a SEPARATE, properly-matched set of temperatures
%                                here (not unfoldedParamsPlain) is required: the first
%                                combined run fed a plain-init-trained search a
%                                pooling-informed input at test time only, a train/test
%                                distribution mismatch that made the combination look
%                                like it hurt. See buildCombinedTrainingData.m.
%
% Direction 1's poolNet/mlpNet are FROZEN throughout (never updated,
% used only to produce features/initial estimates). Direction 2's own
% temperatures ARE retrained (twice, separately) to match each arm's
% actual initialization distribution -- this is still consistent with
% the "frozen pooling net" design decision, which was specifically about
% not backpropagating through poolNet itself.
%
% Requires original/deep-las-matlab, direction1_structured_gru/, AND
% direction2_unfolded_las/ all on the MATLAB path.
%
% Inputs:
%   mlpNet, gruNet_orig       - original paper's trained MLP/GRU (baseline (a))
%   poolNet                   - Direction 1's frozen trained pooling net
%   Lseq, N, L, tau            - Direction 1's channel/window parameters
%   unfoldedParamsPlain        - struct with .temps1, .temps2, trained via
%                                direction2_unfolded_las/buildUnfoldedTrainingData.m
%   unfoldedParamsCombined     - struct with .temps1, .temps2, trained via
%                                buildCombinedTrainingData.m (this folder)
%   minErrors, maxBlocksFactor - adaptive stopping, same convention as
%                                the other simulateBER* functions
%
% Output: BER is a struct with fields .a, .b, .c, .d (each 1 x nSNR),
% one BER curve per arm.

if nargin < 13 || isempty(minErrors),       minErrors = 150; end
if nargin < 14 || isempty(maxBlocksFactor), maxBlocksFactor = 40; end

cfg = getConfig();
Nt = cfg.Nt; Nr = cfg.Nr;
qh = qamHelpers();
Es = qh.symEnergy(M);
B = log2(M);
nBitsPerDim = log2(sqrt(M));
[levels, bitTable] = pamBitTable(M);

rawTemps1_c = log(unfoldedParamsPlain.temps1);
rawTemps2_c = log(unfoldedParamsPlain.temps2);
rawTemps1_d = log(unfoldedParamsCombined.temps1);
rawTemps2_d = log(unfoldedParamsCombined.temps2);

if N < Lseq
    error('simulateBERCombinedFourArms:badN', 'N must be >= Lseq.');
end
nSeqPerRealization = floor(N / Lseq);

nSNR = numel(SNRdB_range);
BER.a = zeros(1, nSNR); BER.b = zeros(1, nSNR);
BER.c = zeros(1, nSNR); BER.d = zeros(1, nSNR);

maxBlocks = maxBlocksFactor * nBlocksMin;

for si = 1:nSNR
    snrdB = SNRdB_range(si);
    sigma2 = Es / 10^(snrdB/10);

    nErrA=0; nErrB=0; nErrC=0; nErrD=0; nBits=0; nTargetsRun=0;

    while true
        Hk_all = genMultipathChannel(Nr, Nt, N, L, tau);

        UG_full = zeros(2*Nt+1, N);
        y_full = zeros(Nr, N); H_full = zeros(Nr, Nt, N);
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

            y_full(:,k) = y; H_full(:,:,k) = H;
            trueBits_full(:, k) = reshape(symbolsToBits(x, M).', [], 1);
        end

        for s = 1:nSeqPerRealization
            idxRange = (s-1)*Lseq + 1 : s*Lseq;
            targetIdx = idxRange(end);
            contextIdx = idxRange(1:end-1);

            y_t = y_full(:, targetIdx);
            H_t = H_full(:, :, targetIdx);
            trueBitsTarget = reshape(trueBits_full(:, targetIdx), B, Nt).';

            % --- (a) Original Deep LAS ---
            LLR_a = deepLASPredict(y_t, H_t, sigma2, M, mlpNet, gruNet_orig);
            hardA = double(LLR_a < 0);

            % --- (b) Direction-1-only: pooling ---
            targetFeat = UG_full(:, targetIdx);
            contextFeat = UG_full(:, contextIdx);
            pooled = [targetFeat; mean(contextFeat,2); max(contextFeat,[],2); std(contextFeat,0,2)];
            llrRow_b = predict(poolNet, pooled.');
            LLR_b = reshape(llrRow_b(:), B, Nt).';
            hardB = double(LLR_b < 0);

            % --- (c) Direction-2-only: plain single-subcarrier hard init,
            % temperatures trained on THIS SAME distribution ---
            xhat0_plain = initEstimate(y_t, H_t, sigma2, M, 'mmse');
            xr0_plain = [real(xhat0_plain); imag(xhat0_plain)];
            LLR_c = unfoldedSoftLAS_B_Soft(y_t, H_t, sigma2, M, xr0_plain, rawTemps1_c, rawTemps2_c);
            LLR_c = extractIfDlLocal(LLR_c);
            hardC = double(LLR_c < 0);

            % --- (d) COMBINED: pooling-informed init, temperatures
            % trained on THIS SAME (pooling-informed) distribution ---
            xr0_combined = pooledLLRToInitEstimate(LLR_b, Nt, nBitsPerDim, levels, bitTable);
            LLR_d = unfoldedSoftLAS_B_Soft(y_t, H_t, sigma2, M, xr0_combined, rawTemps1_d, rawTemps2_d);
            LLR_d = extractIfDlLocal(LLR_d);
            hardD = double(LLR_d < 0);

            nErrA = nErrA + sum(sum(hardA ~= trueBitsTarget));
            nErrB = nErrB + sum(sum(hardB ~= trueBitsTarget));
            nErrC = nErrC + sum(sum(hardC ~= trueBitsTarget));
            nErrD = nErrD + sum(sum(hardD ~= trueBitsTarget));
            nBits = nBits + numel(trueBitsTarget);
            nTargetsRun = nTargetsRun + 1;
        end

        % Stop once the SLOWEST-to-converge arm (typically the smallest
        % errors, usually (d) or (c) at high SNR) has enough errors, or
        % the block cap is hit -- matches the other simulateBER*
        % functions' adaptive-stopping convention, applied to the
        % minimum error count across all four arms.
        if min([nErrA nErrB nErrC nErrD]) >= minErrors || nTargetsRun >= maxBlocks
            break;
        end
    end

    BER.a(si) = nErrA/nBits; BER.b(si) = nErrB/nBits;
    BER.c(si) = nErrC/nBits; BER.d(si) = nErrD/nBits;
    fprintf('[combined, %d-QAM] SNR=%2d dB -> a(orig)=%.3e b(D1)=%.3e c(D2)=%.3e d(combined)=%.3e (%d targets)\n', ...
        M, snrdB, BER.a(si), BER.b(si), BER.c(si), BER.d(si), nTargetsRun);
end
end

function xr0 = pooledLLRToInitEstimate(LLR_pooled, Nt, nBitsPerDim, levels, bitTable)
% Maps the pooling net's hard-decided bits back to the nearest PAM level
% per real dimension, giving a context-informed initial estimate for
% Direction 2's unfolded search -- the actual "feed the structured input
% into the unfolded architecture" combination point.
hardBits = double(LLR_pooled < 0);   % Nt x B, same layout as symbolsToBits.m
Nt2 = 2*Nt;
xr0 = zeros(Nt2, 1);
for n = 1:Nt2
    if n <= Nt
        bitsRow = hardBits(n, 1:nBitsPerDim);          % I-part, antenna n
    else
        bitsRow = hardBits(n-Nt, nBitsPerDim+1:2*nBitsPerDim);   % Q-part, antenna n-Nt
    end
    matchIdx = find(all(bitTable == bitsRow, 2), 1);
    xr0(n) = levels(matchIdx);
end
end

function v = extractIfDlLocal(v)
if isa(v, 'dlarray')
    v = extractdata(v);
end
end
