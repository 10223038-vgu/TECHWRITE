function trainData = buildCombinedTrainingData(M, SNRdB_list, nSamples, mlpNet, poolNet, Lseq, N, L, tau)
% BUILDCOMBINEDTRAININGDATA Same role as
% direction2_unfolded_las/buildUnfoldedTrainingData.m, but xr0 comes from
% the FROZEN Direction 1 pooling net's hard-decided context-informed
% estimate instead of the plain single-subcarrier hard MMSE estimate.
%
% *** WHY THIS EXISTS ***
% The first combined run (poolNet-informed init fed into Direction 2's
% Option A search at TEST time only, with temperatures trained on
% plain-init data) showed the combined system consistently, mildly
% UNDERPERFORMING Direction-2-only -- a train/test distribution mismatch:
% the unfolded search's learned temperatures were never validated on the
% initialization distribution they were actually being asked to refine
% at test time. This builder produces training data that matches the
% ACTUAL test-time distribution, so Direction 2's temperatures can be
% retrained to properly handle pooling-informed starting points. The
% pooling net itself remains frozen throughout (only used to produce
% xr0, never updated) -- consistent with the "frozen pooling net"
% design decision.
%
% Uses correlated multipath channels (matching Direction 1's own
% training distribution), NOT i.i.d. flat channels -- necessary since a
% pooling-informed estimate requires neighboring correlated subcarriers
% to exist at all.
%
% Requires original/deep-las-matlab, direction1_structured_gru/, and
% direction2_unfolded_las/ all on the MATLAB path.

cfg = getConfig();
Nt = cfg.Nt; Nr = cfg.Nr;
qh = qamHelpers();
Es = qh.symEnergy(M);
B = log2(M);
nBitsPerDim = log2(sqrt(M));
[levels, bitTable] = pamBitTable(M);
nSNR = numel(SNRdB_list);

if N < Lseq
    error('buildCombinedTrainingData:badN', 'N must be >= Lseq.');
end
nSeqPerRealization = floor(N / Lseq);

trainData = struct('y', {}, 'H', {}, 'sigma2', {}, 'M', {}, 'xr0', {}, 'trueBits', {});
trainData(nSamples).y = [];

sampleCount = 0;
while sampleCount < nSamples
    snrdB = SNRdB_list(mod(sampleCount, nSNR) + 1);
    sigma2 = Es / 10^(snrdB/10);

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
        if sampleCount >= nSamples, break; end
        idxRange = (s-1)*Lseq + 1 : s*Lseq;
        targetIdx = idxRange(end);
        contextIdx = idxRange(1:end-1);

        targetFeat = UG_full(:, targetIdx);
        contextFeat = UG_full(:, contextIdx);
        pooled = [targetFeat; mean(contextFeat,2); max(contextFeat,[],2); std(contextFeat,0,2)];
        llrRow = predict(poolNet, pooled.');
        LLR_pooled = reshape(llrRow(:), B, Nt).';
        hardBits = double(LLR_pooled < 0);

        Nt2 = 2*Nt;
        xr0 = zeros(Nt2, 1);
        for dnum = 1:Nt2
            if dnum <= Nt
                bitsRow = hardBits(dnum, 1:nBitsPerDim);
            else
                bitsRow = hardBits(dnum-Nt, nBitsPerDim+1:2*nBitsPerDim);
            end
            matchIdx = find(all(bitTable == bitsRow, 2), 1);
            xr0(dnum) = levels(matchIdx);
        end

        sampleCount = sampleCount + 1;
        trainData(sampleCount).y = y_full(:, targetIdx);
        trainData(sampleCount).H = H_full(:, :, targetIdx);
        trainData(sampleCount).sigma2 = sigma2;
        trainData(sampleCount).M = M;
        trainData(sampleCount).xr0 = xr0;
        trainData(sampleCount).trueBits = reshape(trueBits_full(:, targetIdx), B, Nt).';
    end
end
end
