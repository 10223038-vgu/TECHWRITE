function [xr_final, F0, z_final] = unfoldedLAS1(yr, Hr, xr0, M, alphas1)
% UNFOLDEDLAS1 Unfolded version of las1Hard.m (Algorithm 1): identical
% full-PAM-grid candidate structure, but K=numel(alphas1) learnable-
% step-size layers via unfoldedLASCore.m instead of a variable-length
% classical search. Returns the final likelihood cost F0 (Eq. 7) computed
% the same way as lasSearchCore.m, so it stays comparable to the
% classical F0 and reusable in the classical LLR combination formula
% (Unfolded-B, see unfoldedSoftLAS.m).

qh = qamHelpers();
fullLvl = qh.pamLevels(M);
Nt2 = numel(xr0);
candSets = repmat({fullLvl}, 1, Nt2);

[xr_final, z_final] = unfoldedLASCore(yr, Hr, xr0, candSets, alphas1);

Q = Hr' * Hr;
F0 = xr_final' * Q * xr_final - 2*yr'*Hr*xr_final;   % Eq. 7, same formula as lasSearchCore.m
end
