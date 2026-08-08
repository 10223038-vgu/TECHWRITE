function LLR = unfoldedSoftLAS_B_Soft(y, H, sigma2, M, xr0, rawTemps1, rawTemps2)
% UNFOLDEDSOFTLAS_B_SOFT Option A version of unfoldedSoftLAS_B.m: both
% Algorithm 1 (unfoldedLAS1Soft.m) and Algorithm 2
% (unfoldedModLASCounterSoft.m) use the fully differentiable softmax
% relaxation instead of Option B's detached hard decision + learnable
% step size. Combined via the same classical (F1-F0)/sigma^2 formula.
%
% Inputs mirror unfoldedSoftLAS_B.m exactly, except alphas1/alphas2 (step
% sizes) are replaced by rawTemps1/rawTemps2 (unconstrained log-temperature
% parameters, see unfoldedLASCoreSoft.m).

Nt = size(H, 2);
B = log2(M);
nBitsPerDim = log2(sqrt(M));

[Hr, yr] = complexToReal(H, y);
Nt2 = 2*Nt;

[xr_hat, F0_total] = unfoldedLAS1Soft(yr, Hr, xr0, M, rawTemps1);

[levels, bitTable] = pamBitTable(M);
LLR_dim = cell(Nt2, nBitsPerDim);

for n = 1:Nt2
    curLevel = extractIfDl(xr_hat(n));
    [~, curIdx] = min(abs(levels - curLevel));
    for j = 1:nBitsPerDim
        detectedBit = bitTable(curIdx, j);
        F1 = unfoldedModLASCounterSoft(yr, Hr, xr_hat, n, j, M, rawTemps2);

        if detectedBit == 0
            minA0 = F0_total; minA1 = F1;
        else
            minA1 = F0_total; minA0 = F1;
        end
        LLR_dim{n, j} = (1/sigma2) * (minA1 - minA0);
    end
end

rows = cell(Nt2, 1);
for n = 1:Nt2
    rows{n} = horzcat(LLR_dim{n, :});
end
LLR_dim_mat = vertcat(rows{:});

LLR_rows = cell(Nt, 1);
for k = 1:Nt
    LLR_rows{k} = horzcat(LLR_dim_mat(k, :), LLR_dim_mat(Nt + k, :));
end
LLR = vertcat(LLR_rows{:});
end

function v = extractIfDl(v)
if isa(v, 'dlarray')
    v = extractdata(v);
end
end
