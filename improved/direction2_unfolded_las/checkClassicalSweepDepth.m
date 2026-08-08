function checkClassicalSweepDepth(M, SNRdB_range, nSamplesPerSNR)
% CHECKCLASSICALSWEEPDEPTH Measures how many outer sweeps classical
% Algorithm 1 (and one representative Algorithm 2 counter search) need to
% converge, across the SNR range you'll actually train/evaluate on. This
% is a cheap, no-training diagnostic to set K1/K2 in
% run_direction2_vs_original.m to a MEASURED value instead of a guess.
%
% Does NOT modify lasSearchCore.m/las1Hard.m/modLASCounter.m (keeps the
% original reproduction untouched) -- reimplements the same loop
% structure locally with a sweep counter added, since lasSearchCore.m's
% own nIter output counts accepted per-dimension moves, not outer sweeps.
%
% Requires original/deep-las-matlab on the MATLAB path.

if nargin < 1 || isempty(M), M = 4; end
if nargin < 2 || isempty(SNRdB_range), SNRdB_range = 0:2:14; end
if nargin < 3 || isempty(nSamplesPerSNR), nSamplesPerSNR = 200; end

cfg = getConfig();
Nt = cfg.Nt; Nr = cfg.Nr;
qh = qamHelpers();
Es = qh.symEnergy(M);
fullLvl = qh.pamLevels(M);
[levels, bitTable] = pamBitTable(M);

fprintf('%-6s %-14s %-14s %-14s %-14s\n', 'SNR', 'Alg1 mean', 'Alg1 max', 'Alg2 mean', 'Alg2 max');
for si = 1:numel(SNRdB_range)
    snrdB = SNRdB_range(si);
    sigma2 = Es / 10^(snrdB/10);

    alg1Sweeps = zeros(nSamplesPerSNR, 1);
    alg2Sweeps = zeros(nSamplesPerSNR, 1);

    for s = 1:nSamplesPerSNR
        H = genChannel(Nr, Nt);
        symIdx = randi([0 M-1], Nt, 1);
        x = qh.mod(symIdx, M);
        n = sqrt(sigma2/2) * (randn(Nr,1) + 1i*randn(Nr,1));
        y = H*x + n;

        [Hr, yr] = complexToReal(H, y);
        % Same fix as buildUnfoldedTrainingData.m/simulateBERUnfoldedLAS.m:
        % must match softOutputLAS.m's own HARD-decided initialization.
        xhat0 = initEstimate(y, H, sigma2, M, 'mmse');
        xr0 = [real(xhat0); imag(xhat0)];
        Nt2 = numel(xr0);

        candSets1 = repmat({fullLvl}, 1, Nt2);
        [xr_hat, nSweeps1] = countedSweepSearch(yr, Hr, xr0, candSets1, cfg.maxLASIter);
        alg1Sweeps(s) = nSweeps1;

        % One representative Algorithm-2 counter search (dim 1, bit 1)
        curLevel = xr_hat(1);
        [~, curIdx] = min(abs(levels - curLevel));
        flippedBitVal = 1 - bitTable(curIdx, 1);
        allowedLevels = levels(bitTable(:,1) == flippedBitVal);
        xr_init2 = xr_hat;
        [~, iSel] = min(abs(allowedLevels - curLevel));
        xr_init2(1) = allowedLevels(iSel);
        candSets2 = candSets1;
        candSets2{1} = allowedLevels;
        [~, nSweeps2] = countedSweepSearch(yr, Hr, xr_init2, candSets2, cfg.maxLASIter);
        alg2Sweeps(s) = nSweeps2;
    end

    fprintf('%-6d %-14.2f %-14d %-14.2f %-14d\n', snrdB, ...
        mean(alg1Sweeps), max(alg1Sweeps), mean(alg2Sweeps), max(alg2Sweeps));
end

fprintf('\nSet K1/K2 in run_direction2_vs_original.m to roughly the "max" values\n');
fprintf('above (or the mean if max looks like a rare outlier) -- that ensures the\n');
fprintf('unfolded network has enough depth to reach what classical LAS reaches,\n');
fprintf('so any remaining gap is attributable to training, not truncated depth.\n');
end

function [xr_hat, nSweeps] = countedSweepSearch(yr, Hr, xr0, candSets, maxIter)
% Same logic as lasSearchCore.m, with an outer-sweep counter added.
Q = Hr' * Hr;
xr = xr0;
z = Hr' * (yr - Hr*xr);
Nt2 = numel(xr);

nSweeps = 0;
for outer = 1:maxIter
    improvedAny = false;
    for n = 1:Nt2
        qnn = Q(n,n);
        zn = z(n);
        levels = candSets{n};
        bestCost = 0; bestLambda = 0;
        for c = levels
            lambda = c - xr(n);
            if lambda == 0, continue; end
            dCost = qnn*lambda^2 - 2*lambda*zn;
            if dCost < bestCost
                bestCost = dCost; bestLambda = lambda;
            end
        end
        if bestLambda ~= 0
            xr(n) = xr(n) + bestLambda;
            z = z - bestLambda*Q(:,n);
            improvedAny = true;
        end
    end
    nSweeps = outer;
    if ~improvedAny
        break;
    end
end
xr_hat = xr;
end
