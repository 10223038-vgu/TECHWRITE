function [xr_hat, F0, nIter] = las1Hard(yr, Hr, xr0, M, maxIter)
% LAS1HARD Conventional single-symbol-update (1-LAS) hard-output
% detector, real-valued domain. Implements the iterative
% likelihood-decreasing local search of Algorithm 1 / Eqs. 7-13,
% via the shared coordinate-descent core in lasSearchCore.m (every
% real dimension is free to move to any level on the full PAM grid).
%
% Inputs:
%   yr    - real-valued received vector (2Nr x 1)
%   Hr    - real-valued channel matrix (2Nr x 2Nt)
%   xr0   - initial real-valued estimate (2Nt x 1), on the PAM grid
%   M     - QAM order (PAM grid per real dimension is sqrt(M)-ary)
%   maxIter - safety cap on outer sweeps
%
% Outputs:
%   xr_hat - detected real-valued vector (2Nt x 1)
%   F0     - final minimized likelihood cost, Eq. 7
%   nIter  - number of accepted updates performed

qh = qamHelpers();
fullLvl = qh.pamLevels(M);
Nt2 = numel(xr0);
candSets = repmat({fullLvl}, 1, Nt2);   % every dim: full grid, Algorithm 1

[xr_hat, F0, nIter] = lasSearchCore(yr, Hr, xr0, candSets, maxIter);
end
