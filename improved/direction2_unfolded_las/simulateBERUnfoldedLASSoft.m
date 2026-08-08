function BER = simulateBERUnfoldedLASSoft(variant, M, SNRdB_range, nBlocksMin, params, minErrors, maxBlocksFactor)
% SIMULATEBERUNFOLDEDLASSOFT BER simulator for the Option A (soft
% relaxation) Unfolded-A/B variants at inference time. Same structure as
% simulateBERUnfoldedLAS.m (Option B).

if nargin < 6 || isempty(minErrors),       minErrors = 150; end
if nargin < 7 || isempty(maxBlocksFactor), maxBlocksFactor = 40; end

cfg = getConfig();
Nt = cfg.Nt; Nr = cfg.Nr;
qh = qamHelpers();
Es = qh.symEnergy(M);
B = log2(M);

rawTemps1 = log(params.temps1);
if strcmpi(variant, 'B')
    rawTemps2 = log(params.temps2);
else
    W_readout = params.W_readout;
    b_readout = params.b_readout;
end

BER = zeros(size(SNRdB_range));
maxBlocks = maxBlocksFactor * nBlocksMin;

for si = 1:numel(SNRdB_range)
    snrdB = SNRdB_range(si);
    sigma2 = Es / 10^(snrdB/10);

    nErr = 0; nBits = 0; nBlocksRun = 0;

    while true
        H = genChannel(Nr, Nt);
        symIdx = randi([0 M-1], Nt, 1);
        x = qh.mod(symIdx, M);
        n = sqrt(sigma2/2) * (randn(Nr,1) + 1i*randn(Nr,1));
        y = H*x + n;

        xhat0 = initEstimate(y, H, sigma2, M, 'mmse');   % matches classical (hard-decided) init
        xr0 = [real(xhat0); imag(xhat0)];
        trueBits = symbolsToBits(x, M);

        if strcmpi(variant, 'B')
            LLR = unfoldedSoftLAS_B_Soft(y, H, sigma2, M, xr0, rawTemps1, rawTemps2);
        else
            LLR = unfoldedSoftLAS_A_Soft(y, H, sigma2, M, xr0, rawTemps1, W_readout, b_readout);
        end
        LLR = extractIfDl(LLR);

        hardBits = double(LLR < 0);
        nErr = nErr + sum(sum(hardBits ~= trueBits));
        nBits = nBits + numel(trueBits);
        nBlocksRun = nBlocksRun + 1;

        if nErr >= minErrors || nBlocksRun >= maxBlocks
            break;
        end
    end

    BER(si) = nErr / nBits;
    fprintf('[unfolded-%s-soft, %d-QAM] SNR=%2d dB -> BER=%.3e (%d errors / %d bits, %d blocks)\n', ...
        variant, M, snrdB, BER(si), nErr, nBits, nBlocksRun);
end
end

function v = extractIfDl(v)
if isa(v, 'dlarray')
    v = extractdata(v);
end
end
