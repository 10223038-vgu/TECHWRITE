function [F1, xr_counter] = unfoldedModLASCounter(yr, Hr, xr_hat, dimIdx, bitIdx, M, alphas2)
% UNFOLDEDMODLASCOUNTER Unfolded version of modLASCounter.m (Algorithm 2):
% identical bit-flip-constrained candidate-set construction (the flipped
% real dimension is restricted to PAM levels sharing the target bit
% value; every other dimension keeps the full grid), but
% K=numel(alphas2) learnable-step-size layers via unfoldedLASCore.m
% instead of a classical search.
%
% xr_hat is the (unfolded) Step-1 detection from unfoldedLAS1.m -- may be
% a dlarray if it came from a differentiable forward pass; the counter
% search's OWN candidate selection still uses detached values internally
% (see unfoldedLASCore.m), consistent with Option B throughout.

qh = qamHelpers();
fullLvl = qh.pamLevels(M);
[levels, bitTable] = pamBitTable(M);

Nt2 = numel(xr_hat);
curLevel = extractIfDlLocal(xr_hat(dimIdx));
curIdx = find(abs(levels - curLevel) < 1e-9, 1);
if isempty(curIdx)
    % Unfolded xr_hat need not land exactly on a PAM grid point (unlike
    % the classical hard decision) -- snap to nearest grid level so the
    % bit-flip logic below stays well-defined.
    [~, curIdx] = min(abs(levels - curLevel));
end

flippedBitVal = 1 - bitTable(curIdx, bitIdx);
allowedMask = bitTable(:, bitIdx) == flippedBitVal;
allowedLevels = levels(allowedMask);

xr_init = xr_hat;
[~, iSel] = min(abs(allowedLevels - curLevel));
% Additive one-hot update instead of indexed assignment (xr_init(dimIdx)
% = ...) -- same reasoning as unfoldedLASCore.m: dlarray does not
% reliably support indexed assignment under automatic differentiation.
% delta is computed from a DETACHED value (curLevel), consistent with
% this whole design's "no gradient through the discrete selection itself"
% approach -- xr_init's gradient w.r.t. upstream parameters flows
% entirely through the xr_hat term.
e_dim = zeros(Nt2, 1);
e_dim(dimIdx) = 1;
delta = allowedLevels(iSel) - curLevel;
xr_init = xr_init + delta * e_dim;

candSets = repmat({fullLvl}, 1, Nt2);
candSets{dimIdx} = allowedLevels;

[xr_counter, ~] = unfoldedLASCore(yr, Hr, xr_init, candSets, alphas2);

Q = Hr' * Hr;
F1 = xr_counter' * Q * xr_counter - 2*yr'*Hr*xr_counter;
end

function v = extractIfDlLocal(v)
if isa(v, 'dlarray')
    v = extractdata(v);
end
end
