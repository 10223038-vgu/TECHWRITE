function [xhat, LLR, F0] = bruteForceSD(y, H, sigma2, M)
% BRUTEFORCESD Exact optimal soft-output MIMO detector via exhaustive
% search over the full constellation, i.e. literally computing
%   xhat = argmin_x ||y - Hx||^2                              (Eq. 3)
%   l_ij = (1/sigma^2) (min_{A_i^1}||y-Hx||^2 - min_{A_i^0}||y-Hx||^2)  (Eq. 4b)
% by evaluating every one of the M^Nt candidate vectors.
%
% This gives EXACTLY the same answer as an optimal soft-output sphere
% decoder (SD just prunes the search tree to avoid checking every
% candidate; the result is identical). For Nt=4 (this package's
% default), M^Nt is at most 16^4 = 65536, which is small enough to
% brute-force directly in a single vectorized pass -- no need to
% implement an actual sphere-decoding tree search.
%
% WARNING: cost grows as M^Nt. Do not call this with more than ~4-5
% antennas at 16-QAM (or ~8 antennas at 4-QAM) without rewriting this
% as a real bounded-radius sphere decoder -- it will simply run out of
% memory/time otherwise. A guard below warns if the combo count is
% large.
%
% Outputs match softOutputLAS.m's layout: xhat (Nt x 1 complex),
% LLR (Nt x log2(M)), F0 (global ML cost).

Nt = size(H, 2);
B = log2(M);
qh = qamHelpers();
[levels, bitTable] = pamBitTable(M);
nBitsPerDim = log2(sqrt(M));

totalCombos = M^Nt;
if totalCombos > 2e6
    warning('bruteForceSD:largeSearch', ...
        'M^Nt = %d candidates -- this will be slow/memory-heavy. Consider a real bounded sphere decoder for this configuration.', totalCombos);
end

% --- enumerate every candidate symbol-index vector (0..M-1 per antenna) ---
idxGrids = cell(1, Nt);
[idxGrids{:}] = ndgrid(0:M-1);
idxMat = zeros(Nt, totalCombos);
for t = 1:Nt
    idxMat(t, :) = idxGrids{t}(:).';
end

Xall = qh.mod(idxMat, M);          % Nt x totalCombos, complex candidates
residual = y - H*Xall;             % Nr x totalCombos
costs = sum(abs(residual).^2, 1);  % 1 x totalCombos  (Eq. 3 for every candidate)

[F0, bestIdx] = min(costs);
xhat = Xall(:, bestIdx);

% --- bit pattern for every candidate, every antenna (vectorized) ---
bitsAll = zeros(Nt*B, totalCombos);
for t = 1:Nt
    reVals = real(Xall(t, :));
    imVals = imag(Xall(t, :));
    [~, reIdx] = ismember(reVals, levels);
    [~, imIdx] = ismember(imVals, levels);
    bitsAll((t-1)*B + 1 : (t-1)*B + nBitsPerDim, :) = bitTable(reIdx, :).';
    bitsAll((t-1)*B + nBitsPerDim + 1 : t*B, :)      = bitTable(imIdx, :).';
end

% --- Eq. 4b per bit ---
llrFlat = zeros(Nt*B, 1);
for r = 1:Nt*B
    mask0 = bitsAll(r, :) == 0;
    mask1 = ~mask0;
    minA0 = min(costs(mask0));
    minA1 = min(costs(mask1));
    llrFlat(r) = (1/sigma2) * (minA1 - minA0);
end

LLR = reshape(llrFlat, B, Nt).';   % Nt x B, matches softOutputLAS.m layout
end
