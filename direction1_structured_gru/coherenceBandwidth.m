function Bc = coherenceBandwidth(N, L)
% COHERENCEBANDWIDTH Approximate coherence bandwidth in subcarriers
% (THEORY.md, Section 1.4): Bc ~= N / L.
%
% Use this to set the GRU sequence length (THEORY.md, Section 1.6):
%   Lseq ~= coherenceBandwidth(N, L)
% so each training/inference sequence spans approximately one
% coherence block.

Bc = N / L;
end
