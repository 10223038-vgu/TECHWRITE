function BER = simulateBERUnfoldedLAS(variant, M, SNRdB_range, nBlocksMin, params, minErrors, maxBlocksFactor)
% SIMULATEBERUNFOLDEDLAS BER simulator for Unfolded-A / Unfolded-B, at
% inference time (learned parameters held fixed, no gradients needed --
% params are plain numeric, not dlarray, coming out of
% trainUnfoldedLAS.m's output). Same adaptive-stopping style as
% simulateBERPooling.m / simulateBERDirection1.m for direct comparability.
%
% Inputs:
%   variant                    - 'A' or 'B'
%   M, SNRdB_range, nBlocksMin - as in the other simulateBER* functions
%   params                     - struct from trainUnfoldedLAS.m (fields
%                                depend on variant, see that file)
%   minErrors, maxBlocksFactor - adaptive-stopping controls (defaults
%                                150 / 40, same as simulateBERDirection1.m)

if nargin < 6 || isempty(minErrors),       minErrors = 150; end
if nargin < 7 || isempty(maxBlocksFactor), maxBlocksFactor = 40; end

cfg = getConfig();
Nt = cfg.Nt; Nr = cfg.Nr;
qh = qamHelpers();
Es = qh.symEnergy(M);
B = log2(M);

alphas1 = params.alphas1;
if strcmpi(variant, 'B')
    alphas2 = params.alphas2;
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

        % See buildUnfoldedTrainingData.m's comment: LAS's own search
        % must start from the HARD-decided estimate, not the soft one.
        xhat0 = initEstimate(y, H, sigma2, M, 'mmse');
        xr0 = [real(xhat0); imag(xhat0)];
        trueBits = symbolsToBits(x, M);   % Nt x B

        if strcmpi(variant, 'B')
            LLR = unfoldedSoftLAS_B(y, H, sigma2, M, xr0, alphas1, alphas2);
        else
            LLR = unfoldedSoftLAS_A(y, H, sigma2, M, xr0, alphas1, W_readout, b_readout);
        end
        LLR = extractIfDl(LLR);   % plain numeric at inference time

        hardBits = double(LLR < 0);   % same convention as the rest of the codebase
        nErr = nErr + sum(sum(hardBits ~= trueBits));
        nBits = nBits + numel(trueBits);
        nBlocksRun = nBlocksRun + 1;

        if nErr >= minErrors || nBlocksRun >= maxBlocks
            break;
        end
    end

    BER(si) = nErr / nBits;
    fprintf('[unfolded-%s, %d-QAM] SNR=%2d dB -> BER=%.3e (%d errors / %d bits, %d blocks)\n', ...
        variant, M, snrdB, BER(si), nErr, nBits, nBlocksRun);
end
end

function v = extractIfDl(v)
if isa(v, 'dlarray')
    v = extractdata(v);
end
end
