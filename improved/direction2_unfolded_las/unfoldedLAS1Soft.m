function [xr_final, F0, z_final] = unfoldedLAS1Soft(yr, Hr, xr0, M, rawTemps1)
% UNFOLDEDLAS1SOFT Soft-relaxation (Option A) version of unfoldedLAS1.m:
% same full-PAM-grid candidate structure, but via unfoldedLASCoreSoft.m's
% fully differentiable softmax relaxation instead of a detached hard
% decision. F0 is computed with the same closed-form cost formula as the
% classical algorithm, for direct comparability.

qh = qamHelpers();
fullLvl = qh.pamLevels(M);
Nt2 = numel(xr0);
candSets = repmat({fullLvl}, 1, Nt2);

[xr_final, z_final] = unfoldedLASCoreSoft(yr, Hr, xr0, candSets, rawTemps1);

Q = Hr' * Hr;
F0 = xr_final' * Q * xr_final - 2*yr'*Hr*xr_final;
end
