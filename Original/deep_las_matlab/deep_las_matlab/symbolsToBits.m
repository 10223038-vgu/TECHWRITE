function bitsMat = symbolsToBits(x, M)
% SYMBOLSTOBITS Convert a complex Nt x 1 QAM symbol vector into an
% Nt x log2(M) bit matrix using the same per-dimension Gray-coded
% PAM decomposition used throughout the detector code
% ([I-bits, Q-bits] per row), so it lines up 1:1 with the LLR
% matrices produced by softOutputLAS.m / deepLASPredict.m.

[levels, bitTable] = pamBitTable(M);
Nt = numel(x);
nBitsPerDim = log2(sqrt(M));
bitsMat = zeros(Nt, 2*nBitsPerDim);

for k = 1:Nt
    reIdx = find(abs(levels - real(x(k))) < 1e-9, 1);
    imIdx = find(abs(levels - imag(x(k))) < 1e-9, 1);
    bitsMat(k, :) = [bitTable(reIdx, :), bitTable(imIdx, :)];
end
end
