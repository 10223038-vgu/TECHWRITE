function [xhat_complex, LLR, F0_total] = softOutputLAS(y, H, sigma2, M, initType, maxIter)
% SOFTOUTPUTLAS Model-based two-step soft-output LAS detector
% (Section III of the paper).
%
% Step 1: detect xhat via conventional 1-LAS (Algorithm 1) -> F0
% Step 2: for every bit, find the best counter-hypothesis via
%         modified-LAS (Algorithm 2) -> F1, then combine per the
%         max-log-MAP LLR formula (Eq. 4b):
%           l_ij = (1/sigma^2) * ( min_{A_i^1} ||y-Hx||^2
%                                 - min_{A_i^0} ||y-Hx||^2 )
%
% Inputs:
%   y        - received complex vector (Nr x 1)
%   H        - complex channel matrix (Nr x Nt)
%   sigma2   - noise variance (per real+imag dimension, as used
%              consistently in initEstimate.m)
%   M        - QAM order (4, 16, ...)
%   initType - 'zf' or 'mmse' (default 'mmse')
%   maxIter  - LAS safety cap on outer sweeps (default 50)
%
% Outputs:
%   xhat_complex - detected complex symbol vector (Nt x 1)
%   LLR          - Nt x log2(M) matrix of LLR values (Eq. 19 layout)
%   F0_total     - final Step-1 likelihood cost (Eq. 7)

if nargin < 5 || isempty(initType), initType = 'mmse'; end
if nargin < 6 || isempty(maxIter),  maxIter  = 50;      end

Nt = size(H, 2);
B  = log2(M);
nBitsPerDim = log2(sqrt(M));

% ---- Step 1: detection (Algorithm 1) ----
xhat0 = initEstimate(y, H, sigma2, M, initType);
[Hr, yr] = complexToReal(H, y);
xr0 = [real(xhat0); imag(xhat0)];

[xr_hat, F0_total, ~] = las1Hard(yr, Hr, xr0, M, maxIter);

% ---- Step 2: per-bit counter hypothesis (Algorithm 2) + LLR (Eq. 4b) ----
[levels, bitTable] = pamBitTable(M);
Nt2 = 2*Nt;
LLR_dim = zeros(Nt2, nBitsPerDim);

for n = 1:Nt2
    curLevel = xr_hat(n);
    curIdx = find(abs(levels - curLevel) < 1e-9, 1);
    for j = 1:nBitsPerDim
        detectedBit = bitTable(curIdx, j);
        F1 = modLASCounter(yr, Hr, xr_hat, n, j, M, maxIter);

        if detectedBit == 0
            minA0 = F0_total; minA1 = F1;
        else
            minA1 = F0_total; minA0 = F1;
        end
        LLR_dim(n, j) = (1/sigma2) * (minA1 - minA0);
    end
end

% ---- Reassemble complex symbols and per-symbol B-bit LLR rows ----
xhat_complex = xr_hat(1:Nt) + 1i*xr_hat(Nt+1:end);

LLR = zeros(Nt, B);
for k = 1:Nt
    LLR(k, :) = [LLR_dim(k, :), LLR_dim(Nt + k, :)];   % [I-bits, Q-bits]
end
end
