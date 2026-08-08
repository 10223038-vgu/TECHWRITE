function LLR = unfoldedSoftLAS_B(y, H, sigma2, M, xr0, alphas1, alphas2)
% UNFOLDEDSOFTLAS_B "Unfolded-B" variant: unfolds BOTH Algorithm 1 (via
% unfoldedLAS1.m) and Algorithm 2 (via unfoldedModLASCounter.m, once per
% real-dimension/bit pair), and combines them with the EXACT SAME
% max-log-MAP formula softOutputLAS.m uses:
%   LLR = (1/sigma2) * (F1 - F0), sign depending on the detected bit
%
% The difference from softOutputLAS.m is entirely in HOW F0/F1 are
% computed (K1/K2 learnable-step-size unfolded layers instead of a
% variable-length classical search) and HOW the whole pipeline is
% trained (end-to-end against true transmitted bits via
% trainUnfoldedLAS.m, not distilled from softOutputLAS.m's own output).
%
% Inputs:
%   y, H, sigma2, M - same convention as softOutputLAS.m
%   xr0             - initial real-valued estimate (2Nt x 1), e.g. from
%                     initEstimateSoft.m (kept as a fixed, non-learnable
%                     starting point, same role as in the classical code)
%   alphas1         - K1 x 1 dlarray, Algorithm-1 unfolded step sizes
%   alphas2         - K2 x 1 dlarray, Algorithm-2 unfolded step sizes
%                     (shared across every (dim,bit) counter search --
%                     one set of K2 learnable scalars total, not one per
%                     bit, to keep the parameter count small)
%
% Output:
%   LLR - Nt x log2(M) matrix (same Eq. 19 layout as softOutputLAS.m),
%   dlarray if alphas1/alphas2 are dlarray (for use inside
%   trainUnfoldedLAS.m's loss function; call extractdata(LLR) at
%   inference time in simulateBERUnfoldedLAS.m).

Nt = size(H, 2);
B = log2(M);
nBitsPerDim = log2(sqrt(M));

[Hr, yr] = complexToReal(H, y);
Nt2 = 2*Nt;

[xr_hat, F0_total] = unfoldedLAS1(yr, Hr, xr0, M, alphas1);

[levels, bitTable] = pamBitTable(M);
LLR_dim = cell(Nt2, nBitsPerDim);   % cell so we can hold dlarray scalars cleanly

for n = 1:Nt2
    curLevel = extractIfDl(xr_hat(n));
    [~, curIdx] = min(abs(levels - curLevel));   % snap to nearest grid point
    for j = 1:nBitsPerDim
        detectedBit = bitTable(curIdx, j);
        F1 = unfoldedModLASCounter(yr, Hr, xr_hat, n, j, M, alphas2);

        if detectedBit == 0
            minA0 = F0_total; minA1 = F1;
        else
            minA1 = F0_total; minA0 = F1;
        end
        LLR_dim{n, j} = (1/sigma2) * (minA1 - minA0);
    end
end

% Assemble via horzcat/vertcat (not indexed assignment) -- this is the
% robust way to build a dlarray matrix from individual dlarray scalars
% under automatic differentiation.
rows = cell(Nt2, 1);
for n = 1:Nt2
    rows{n} = horzcat(LLR_dim{n, :});   % 1 x nBitsPerDim
end
LLR_dim_mat = vertcat(rows{:});          % Nt2 x nBitsPerDim

LLR_rows = cell(Nt, 1);
for k = 1:Nt
    LLR_rows{k} = horzcat(LLR_dim_mat(k, :), LLR_dim_mat(Nt + k, :));   % [I-bits, Q-bits]
end
LLR = vertcat(LLR_rows{:});   % Nt x B
end

function v = extractIfDl(v)
if isa(v, 'dlarray')
    v = extractdata(v);
end
end
