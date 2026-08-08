function seqIID = buildIIDSequences(M, SNRdB_list, Lseq, nSequences, mlpNet)
% BUILDIIDSEQUENCES Baseline arm for the Section 1.7 ablation: every
% timestep (including the target) is drawn from an INDEPENDENT
% single-subcarrier i.i.d. Rayleigh channel draw, exactly matching the
% original codebase's arbitrary-pooling approach (original/deep-las-matlab's
% trainGRU.m with seqLen>1). This is the "no correlation structure at all"
% control against which both buildCorrelatedAndShuffledSequences.m arms
% are compared.
%
% SNRdB_list - SNR (dB) VALUE(S) for this dataset. Pass a VECTOR spanning
% the SNR range you'll evaluate BER over (sequences cycle through it
% round-robin), for the same reason given in
% buildCorrelatedAndShuffledSequences.m -- training at one fixed SNR and
% testing across a sweep is an out-of-distribution mismatch. A scalar is
% still accepted for backward compatibility.
%
% Requires original/deep-las-matlab on the MATLAB path (genChannel,
% qamHelpers, getConfig, initEstimateSoft, softOutputLAS).

cfg = getConfig();
Nt = cfg.Nt; Nr = cfg.Nr;
qh = qamHelpers();
Es = qh.symEnergy(M);
B = log2(M);

X = cell(nSequences, 1);
Y = zeros(nSequences, Nt*B);

nSNR = numel(SNRdB_list);
for s = 1:nSequences
    snrdB = SNRdB_list(mod(s-1, nSNR) + 1);   % round-robin through the SNR list
    sigma2 = Es / 10^(snrdB/10);

    UG_seq = zeros(2*Nt+1, Lseq);
    for t = 1:Lseq
        H = genChannel(Nr, Nt);   % fresh independent i.i.d. draw every timestep
        symIdx = randi([0 M-1], Nt, 1);
        x = qh.mod(symIdx, M);
        n = sqrt(sigma2/2) * (randn(Nr,1) + 1i*randn(Nr,1));
        y = H*x + n;

        xhatSoft = initEstimateSoft(y, H, sigma2, M, 'mmse');
        xn = [real(xhatSoft); imag(xhatSoft)];
        xn = xn ./ max(abs(xn));
        llrRough = mean(mlpNet(xn));
        UG_seq(:, t) = [xn; llrRough];

        if t == Lseq   % target = last timestep, same convention as the paired builder
            [~, LLRtrue] = softOutputLAS(y, H, sigma2, M, 'mmse', cfg.maxLASIter);
            Y(s, :) = reshape(LLRtrue.', 1, []);
        end
    end
    X{s} = UG_seq;
end

seqIID.X = X; seqIID.Y = Y;
end
