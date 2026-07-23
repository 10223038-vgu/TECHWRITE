function hardBits = detNetPredict(y, H, detNet)
% DETNETPREDICT Run a trained DetNet (from trainDetNet.m) on one
% received vector and return HARD bit decisions. DetNet, as
% originally proposed, is a hard-output detector (see paper's
% Section I-A: prior DNN-based detectors including DetNet "employ
% hard output detection"), so unlike the other detectors in this
% package it does not produce a genuine soft LLR -- simulateBER.m's
% 'detnet' case uses these hard bits directly rather than
% thresholding a synthetic LLR.

Nt = detNet.Nt;
Nt2 = 2*Nt;
K = detNet.K;
vSize = detNet.vSize;
M = detNet.M;

[Hr, yr] = complexToReal(H, y);
paramsD = extractParamsDouble(detNet.params, K);

xAll = detNetForwardDouble(paramsD, Hr, yr, K, Nt2, vSize);
xFinalNorm = xAll{K};
xFinal = xFinalNorm * detNet.scaleFactor;   % rescale [-1,1] -> PAM range

qh = qamHelpers();
lvl = qh.pamLevels(M);
xq = zeros(size(xFinal));
for i = 1:numel(xFinal)
    [~, idx] = min(abs(lvl - xFinal(i)));
    xq(i) = lvl(idx);
end

xhatComplex = xq(1:Nt) + 1i*xq(Nt+1:end);
hardBits = symbolsToBits(xhatComplex, M);
end

% ==================================================================
function paramsD = extractParamsDouble(params, K)
paramsD = struct();
fields = {'W1','b1','W2','b2','t','delta1','delta2'};
for f = 1:numel(fields)
    fn = fields{f};
    for k = 1:K
        paramsD.(fn){k} = double(extractdata(params.(fn){k}));
    end
end
end

% ==================================================================
function xAll = detNetForwardDouble(params, Hr, yr, K, Nt2, vSize)
HtH = Hr' * Hr;
Hty = Hr' * yr;
x = zeros(Nt2, 1);
v = zeros(vSize, 1);
xAll = cell(K,1);
for k = 1:K
    z = x - params.delta1{k}*Hty + params.delta2{k}*(HtH*x);
    concatIn = [z; x; v];
    h = max(params.W1{k}*concatIn + params.b1{k}, 0);
    outFull = params.W2{k}*h + params.b2{k};
    xRaw = outFull(1:Nt2);
    v = outFull(Nt2+1:end);
    tk = abs(params.t{k}) + 1e-3;
    x = -1 + max(xRaw+tk,0)/tk - max(xRaw-tk,0)/tk;
    xAll{k} = x;
end
end
