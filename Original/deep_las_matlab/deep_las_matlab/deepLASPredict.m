function LLR = deepLASPredict(y, H, sigma2, M, mlpNet, gruNet)
% DEEPLASPREDICT Online inference for the trained Deep LAS detector
% (Eq. 23): estimate LLR directly from the received equalized
% signal, without running the model-based LAS search.
%
% Inputs:
%   y, H, sigma2, M - as elsewhere
%   mlpNet, gruNet  - trained networks from trainMLP.m / trainGRU.m
%
% Output:
%   LLR - Nt x log2(M) LLR matrix (same layout as softOutputLAS.m)

Nt = size(H, 2);
B  = log2(M);

xhat_mmse = initEstimate(y, H, sigma2, M, 'mmse');
xm = [real(xhat_mmse); imag(xhat_mmse)];
xn = xm ./ max(abs(xm));

llrRough = mlpNet(xn);
llrRoughSummary = mean(llrRough);

UG = [xn; llrRoughSummary];
llrRow = predict(gruNet, {UG});   % 1 x (Nt*B) numeric row vector
llrVec = llrRow(:);               % Nt*B x 1

LLR = reshape(llrVec, B, Nt).';  % back to Nt x B, matches softOutputLAS.m layout
end
