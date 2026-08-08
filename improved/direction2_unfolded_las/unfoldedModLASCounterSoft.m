function [F1, xr_counter] = unfoldedModLASCounterSoft(yr, Hr, xr_hat, dimIdx, bitIdx, M, rawTemps2)
% UNFOLDEDMODLASCOUNTERSOFT Soft-relaxation (Option A) version of
% unfoldedModLASCounter.m: same bit-flip-constrained candidate-set
% construction, but the counter search itself runs through
% unfoldedLASCoreSoft.m's differentiable softmax relaxation.
%
% NOTE: determining WHICH bit xr_hat currently represents (needed to
% know which levels to restrict the flipped dimension to) still uses a
% detached nearest-grid-point snap, same as unfoldedModLASCounter.m --
% this is bookkeeping (a discrete labeling decision), not the search
% itself, and making it soft too would require reformulating the entire
% LLR combination as a probability-weighted mixture rather than a single
% branch. Left as a documented simplification; the actual candidate
% search -- the part Option A specifically targets -- is fully soft.

qh = qamHelpers();
fullLvl = qh.pamLevels(M);
[levels, bitTable] = pamBitTable(M);

Nt2 = numel(xr_hat);
curLevel = extractIfDlLocal(xr_hat(dimIdx));
[~, curIdx] = min(abs(levels - curLevel));   % nearest-grid snap (detached, bookkeeping only)

flippedBitVal = 1 - bitTable(curIdx, bitIdx);
allowedMask = bitTable(:, bitIdx) == flippedBitVal;
allowedLevels = levels(allowedMask);

xr_init = xr_hat;
[~, iSel] = min(abs(allowedLevels - curLevel));
e_dim = zeros(Nt2, 1);
e_dim(dimIdx) = 1;
delta = allowedLevels(iSel) - curLevel;
xr_init = xr_init + delta * e_dim;

candSets = repmat({fullLvl}, 1, Nt2);
candSets{dimIdx} = allowedLevels;

[xr_counter, ~] = unfoldedLASCoreSoft(yr, Hr, xr_init, candSets, rawTemps2);

Q = Hr' * Hr;
F1 = xr_counter' * Q * xr_counter - 2*yr'*Hr*xr_counter;
end

function v = extractIfDlLocal(v)
if isa(v, 'dlarray')
    v = extractdata(v);
end
end
