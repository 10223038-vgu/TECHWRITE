function [F1, xr_counter] = modLASCounter(yr, Hr, xr_hat, dimIdx, bitIdx, M, maxIter)
% MODLASCOUNTER Algorithm 2 (modified-LAS): find the best
% counter-hypothesis for one bit of the detected symbol vector.
%
% "Counter symbols" = symbols with the bit of interest flipped
% relative to the detected symbol (Section III-2). The search space
% for the flipped real dimension is restricted to PAM levels that
% share the flipped bit value; every other real dimension remains
% free to move on the full grid ("remaining symbols ... updated
% according to the constellation grid").
%
% Inputs:
%   yr, Hr   - real-valued received vector / channel matrix
%   xr_hat   - detected real-valued vector from Step 1 (2Nt x 1)
%   dimIdx   - which real dimension (1..2*Nt) holds the bit of interest
%   bitIdx   - which bit (1..log2(sqrt(M))) within that dimension's
%              PAM symbol
%   M        - QAM order
%   maxIter  - safety cap on outer sweeps
%
% Outputs:
%   F1          - minimized counter-hypothesis cost, Eq. 14/18
%   xr_counter  - the counter-hypothesis vector found

qh = qamHelpers();
fullLvl = qh.pamLevels(M);
[levels, bitTable] = pamBitTable(M);

Nt2 = numel(xr_hat);
curLevel = xr_hat(dimIdx);
curIdx = find(abs(levels - curLevel) < 1e-9, 1);
if isempty(curIdx)
    error('modLASCounter:badLevel', ...
        'Detected level %.3f is not on the PAM grid for M=%d.', curLevel, M);
end

flippedBitVal = 1 - bitTable(curIdx, bitIdx);
allowedMask = bitTable(:, bitIdx) == flippedBitVal;
allowedLevels = levels(allowedMask);   % counter-symbol subset for this bit

% Initial counter estimate: flip just this bit, keep everything else
xr_init = xr_hat;
% pick the counter level closest to the current one as the initial guess
[~, iSel] = min(abs(allowedLevels - curLevel));
xr_init(dimIdx) = allowedLevels(iSel);

candSets = repmat({fullLvl}, 1, Nt2);
candSets{dimIdx} = allowedLevels;      % constrained dimension

[xr_counter, F1] = lasSearchCore(yr, Hr, xr_init, candSets, maxIter);
end
