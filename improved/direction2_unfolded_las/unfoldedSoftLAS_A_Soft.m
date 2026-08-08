function LLR = unfoldedSoftLAS_A_Soft(y, H, sigma2, M, xr0, rawTemps1, W_readout, b_readout)
% UNFOLDEDSOFTLAS_A_SOFT Option A version of unfoldedSoftLAS_A.m:
% Algorithm 1 only, via the fully differentiable softmax relaxation
% (unfoldedLAS1Soft.m), then the same learned linear readout mapping
% [xr_final; z_final] to per-bit LLRs.

Nt = size(H, 2);
B = log2(M);

[Hr, yr] = complexToReal(H, y);
qh = qamHelpers();
fullLvl = qh.pamLevels(M);
Nt2 = numel(xr0);
candSets = repmat({fullLvl}, 1, Nt2);
[xr_final, z_final] = unfoldedLASCoreSoft(yr, Hr, xr0, candSets, rawTemps1);

feat = [xr_final; z_final];
llrVecFlat = W_readout * feat + b_readout;
LLR = reshape(llrVecFlat, Nt, B);
end
