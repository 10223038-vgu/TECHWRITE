function BER = simulateBERDirection1(M, SNRdB_range, nBlocksMin, mlpNet, gruNet, Lseq, N, L, tau, minErrors, maxBlocksFactor, mode)
% SIMULATEBERDIRECTION1 BER simulator for the Direction-1 (structured-GRU)
% Deep LAS detector, built to match its TRAINING distribution at test time.
%
% *** WHY THIS FILE EXISTS -- THE BUG THIS FIXES ***
% gruNet (from trainGRUFromSequences.m) is trained on MULTI-timestep
% sequences of length Lseq: Lseq-1 context subcarriers followed by the
% target subcarrier, with OutputMode='last' (sequence-to-one -- i.e. it
% produces a genuinely useful hidden state only after being fed real
% context; the trained mapping is "given Lseq timesteps, predict the LLR
% of the last one").
%
% The ORIGINAL codebase's simulateBER.m always routes 'deeplas' through
% deepLASPredict.m, which calls predict(gruNet, {UG}) with a
% SINGLE-timestep UG (a (2Nt+1) x 1 vector). Calling run_direction1_*.m
% with `simulateBER('deeplas', ..., gruNet_d1)` therefore feeds the
% Direction-1 GRU a 1-long sequence it was never trained to produce a
% meaningful "last" output from -- MATLAB does NOT error on this (the
% network happily runs and returns *something*), it just silently
% produces near-random LLRs. That is exactly the flat purple curve stuck
% near BER ~ 0.5 across every SNR: the detector is guessing.
%
% This function instead reconstructs the correctly-shaped Lseq-long
% context per test prediction, mirroring whichever of the three ablation
% arms' training-data construction is selected via `mode`, so the GRU
% sees the same kind of input at test time that it saw in training.
%
% Inputs:
%   M, SNRdB_range, nBlocksMin - as in simulateBER.m ('nBlocksMin' here
%                                counts TARGET subcarriers evaluated, not
%                                whole Nt-symbol blocks, since Direction 1
%                                is evaluated per-subcarrier)
%   mlpNet, gruNet             - trained MLP / Direction-1 GRU
%   Lseq, N, L, tau            - MUST match the values gruNet was trained
%                                with (buildCorrelatedAndShuffledSequences
%                                / buildIIDSequences inputs) -- mismatched
%                                Lseq/N/L/tau here would reintroduce a
%                                train/test mismatch, just a smaller one.
%                                N/L/tau are unused when mode='iid'.
%   minErrors, maxBlocksFactor - adaptive-stopping controls, same
%                                semantics as simulateBER.m (default
%                                150 / 40)
%   mode                       - 'correlated' (default): context in true
%                                consecutive subcarrier order, matches
%                                buildCorrelatedAndShuffledSequences.m's
%                                correlated arm and is what you want for
%                                the actual Direction-1 detector.
%                                'shuffled': identical channel draws/
%                                content, context order randomly
%                                permuted -- matches the shuffled-control
%                                arm, use only when evaluating gruShuf.
%                                'iid': fresh independent i.i.d. channel
%                                draw every timestep -- matches
%                                buildIIDSequences.m, use only when
%                                evaluating gruIID.

if nargin < 10 || isempty(minErrors),       minErrors = 150; end
if nargin < 11 || isempty(maxBlocksFactor), maxBlocksFactor = 40; end
if nargin < 12 || isempty(mode),            mode = 'correlated'; end

cfg = getConfig();
Nt = cfg.Nt; Nr = cfg.Nr;
qh = qamHelpers();
Es = qh.symEnergy(M);
B = log2(M);

if N < Lseq
    error('simulateBERDirection1:badN', 'N must be >= Lseq.');
end
nSeqPerRealization = floor(N / Lseq);
if nSeqPerRealization < 1
    error('simulateBERDirection1:badLseq', 'Lseq must be <= N.');
end

BER = zeros(size(SNRdB_range));
maxBlocks = maxBlocksFactor * nBlocksMin;   % "blocks" = target subcarriers evaluated

for si = 1:numel(SNRdB_range)
    snrdB = SNRdB_range(si);
    sigma2 = Es / 10^(snrdB/10);

    nErr = 0; nBits = 0; nTargetsRun = 0;

    while true
        switch lower(mode)
            case {'correlated', 'shuffled'}
                % One fresh correlated multipath realization per pass, same
                % as buildCorrelatedAndShuffledSequences.m -- gives
                % nSeqPerRealization non-overlapping (context, target)
                % windows per realization.
                Hk_all = genMultipathChannel(Nr, Nt, N, L, tau);

                UG_full = zeros(2*Nt+1, N);
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

                    if strcmpi(mode, 'shuffled')
                        contextIdx = contextIdx(randperm(numel(contextIdx)));
                    end
                    seqIdx = [contextIdx, targetIdx];
                    seqUG = UG_full(:, seqIdx);   % (2Nt+1) x Lseq

                    llrRow = predict(gruNet, {seqUG});   % 1 x (Nt*B)
                    llrVec = llrRow(:);
                    LLR = reshape(llrVec, B, Nt).';       % Nt x B, matches symbolsToBits layout
                    hardBits = double(LLR < 0);

                    trueBitsTarget = reshape(trueBits_full(:, targetIdx), B, Nt).';

                    nErr = nErr + sum(sum(hardBits ~= trueBitsTarget));
                    nBits = nBits + numel(trueBitsTarget);
                    nTargetsRun = nTargetsRun + 1;
                end

            case 'iid'
                % Matches buildIIDSequences.m: every timestep (including
                % the target) is an independent fresh i.i.d. channel draw.
                % Evaluate one Lseq-long sequence per pass; N/L/tau unused.
                UG_seq = zeros(2*Nt+1, Lseq);
                for t = 1:Lseq
                    H = genChannel(Nr, Nt);
                    symIdx = randi([0 M-1], Nt, 1);
                    x = qh.mod(symIdx, M);
                    n = sqrt(sigma2/2) * (randn(Nr,1) + 1i*randn(Nr,1));
                    y = H*x + n;

                    xhatSoft = initEstimateSoft(y, H, sigma2, M, 'mmse');
                    xn = [real(xhatSoft); imag(xhatSoft)];
                    xn = xn ./ max(abs(xn));
                    llrRough = mean(mlpNet(xn));
                    UG_seq(:, t) = [xn; llrRough];

                    if t == Lseq
                        trueBitsTarget = symbolsToBits(x, M);   % Nt x B
                    end
                end

                llrRow = predict(gruNet, {UG_seq});
                llrVec = llrRow(:);
                LLR = reshape(llrVec, B, Nt).';
                hardBits = double(LLR < 0);

                nErr = nErr + sum(sum(hardBits ~= trueBitsTarget));
                nBits = nBits + numel(trueBitsTarget);
                nTargetsRun = nTargetsRun + 1;

            otherwise
                error('simulateBERDirection1:badMode', ...
                    'Unknown mode ''%s'' (expected correlated | shuffled | iid).', mode);
        end

        if nErr >= minErrors || nTargetsRun >= maxBlocks
            break;
        end
    end

    BER(si) = nErr / nBits;
    fprintf('[deeplas-direction1, %d-QAM] SNR=%2d dB -> BER=%.3e (%d errors / %d bits, %d target subcarriers)\n', ...
        M, snrdB, BER(si), nErr, nBits, nTargetsRun);
end
end
