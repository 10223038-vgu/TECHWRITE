function trainData = buildUnfoldedTrainingData(M, SNRdB_list, nSamples)
% BUILDUNFOLDEDTRAININGDATA Generates i.i.d. flat-channel training samples
% for trainUnfoldedLAS.m -- STANDALONE Direction 2 (matches the original
% codebase's channel model exactly, same as generateTrainingData.m; no
% correlated/pooled context from Direction 1 yet, per the confirmed plan
% to combine the two directions only after each works independently).
%
% Each sample gets .y, .H, .sigma2, .M, .xr0 (MMSE soft initial estimate,
% same role as in softOutputLAS.m), and .trueBits (Nt x log2(M), same
% layout as symbolsToBits.m) -- this last field is what makes Direction
% 2's training genuinely "against true bits," unlike the original paper's
% GRU which regressed against softOutputLAS.m's own LLR output.
%
% Requires original/deep-las-matlab on the MATLAB path (genChannel,
% qamHelpers, getConfig, initEstimateSoft, symbolsToBits).

cfg = getConfig();
Nt = cfg.Nt; Nr = cfg.Nr;
qh = qamHelpers();
Es = qh.symEnergy(M);
nSNR = numel(SNRdB_list);

trainData = struct('y', {}, 'H', {}, 'sigma2', {}, 'M', {}, 'xr0', {}, 'trueBits', {});
trainData(nSamples).y = [];   % preallocate

for s = 1:nSamples
    snrdB = SNRdB_list(mod(s-1, nSNR) + 1);
    sigma2 = Es / 10^(snrdB/10);

    H = genChannel(Nr, Nt);
    symIdx = randi([0 M-1], Nt, 1);
    x = qh.mod(symIdx, M);
    n = sqrt(sigma2/2) * (randn(Nr,1) + 1i*randn(Nr,1));
    y = H*x + n;

    % IMPORTANT: LAS's own discrete search must start from the SAME
    % HARD-decided estimate softOutputLAS.m itself uses (initEstimate.m,
    % not initEstimateSoft.m) -- starting from a continuous/off-grid
    % point lets a dimension get permanently "stuck" off-grid (no
    % discrete move looks like an improvement relative to an already-
    % near-optimal continuous point), which silently breaks the discrete
    % search almost entirely. initEstimateSoft.m is the right choice for
    % feeding a neural net (Direction 1), but wrong here.
    xhat0 = initEstimate(y, H, sigma2, M, 'mmse');
    xr0 = [real(xhat0); imag(xhat0)];

    trainData(s).y = y;
    trainData(s).H = H;
    trainData(s).sigma2 = sigma2;
    trainData(s).M = M;
    trainData(s).xr0 = xr0;
    trainData(s).trueBits = symbolsToBits(x, M);   % Nt x log2(M)
end
end
