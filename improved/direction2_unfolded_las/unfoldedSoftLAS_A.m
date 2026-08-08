function LLR = unfoldedSoftLAS_A(y, H, sigma2, M, xr0, alphas1, W_readout, b_readout)
% UNFOLDEDSOFTLAS_A "Unfolded-A" variant: unfolds ONLY Algorithm 1
% (unfoldedLAS1.m), then a small LEARNED LINEAR READOUT maps the final
% state [xr_final; z_final] directly to per-bit LLRs, bypassing
% Algorithm 2 / the classical (F1-F0)/sigma2 combination formula
% entirely. Simpler and cheaper than Unfolded-B (no per-bit counter
% search), and a genuinely different design choice: it lets training
% figure out how to turn the Step-1 state into LLRs, rather than
% presupposing the classical two-step structure is the right one.
%
% Inputs:
%   y, H, sigma2, M - same convention as softOutputLAS.m
%   xr0             - initial real-valued estimate (2Nt x 1), fixed
%                     (non-learnable) starting point
%   alphas1         - K1 x 1 dlarray, Algorithm-1 unfolded step sizes
%   W_readout       - (Nt*log2(M)) x (4*Nt) dlarray, learned readout
%                     weights (see below for the 4*Nt-dim input feature)
%   b_readout       - (Nt*log2(M)) x 1 dlarray, learned readout bias
%
% Output:
%   LLR - Nt x log2(M) matrix (same Eq. 19 layout as softOutputLAS.m)

Nt = size(H, 2);
B = log2(M);

[Hr, yr] = complexToReal(H, y);
[xr_final, z_final] = unfoldedLAS1Internal(yr, Hr, xr0, M, alphas1);

% Readout input: final real-valued state AND the residual correlation
% vector z (a natural "how confident/uncorrected is this dimension"
% signal the classical algorithm already computes internally, but never
% exposes to a learned head) -- 2*(2Nt) = 4*Nt features total.
feat = [xr_final; z_final];   % (4*Nt) x 1 (since numel(xr_final)=numel(z_final)=2*Nt)

llrVecFlat = W_readout * feat + b_readout;   % (Nt*B) x 1, dlarray
LLR = reshape(llrVecFlat, Nt, B);            % matches symbolsToBits.m's Nt x B layout
end

function [xr_final, z_final] = unfoldedLAS1Internal(yr, Hr, xr0, M, alphas1)
qh = qamHelpers();
fullLvl = qh.pamLevels(M);
Nt2 = numel(xr0);
candSets = repmat({fullLvl}, 1, Nt2);
[xr_final, z_final] = unfoldedLASCore(yr, Hr, xr0, candSets, alphas1);
end
