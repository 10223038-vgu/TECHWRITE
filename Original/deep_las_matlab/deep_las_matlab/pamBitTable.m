function [levels, bitTable] = pamBitTable(M)
% PAMBITTABLE Build the Gray-coded PAM level <-> bit-pattern lookup
% for one real dimension (I or Q) of a square M-QAM constellation.
%
%   K = sqrt(M) PAM levels, nBits = log2(K) bits per level.
%
% Outputs:
%   levels   - 1 x K vector of PAM amplitudes, e.g. [-3 -1 1 3] for M=16
%   bitTable - K x nBits matrix; bitTable(i,:) is the bit pattern
%              (MSB first) associated with levels(i)

K = sqrt(M);
nBits = log2(K);
idx = (0:K-1)';

levels = pammod(idx, K, 0, 'gray').';
bitTable = de2bi(idx, nBits, 'left-msb');
end
