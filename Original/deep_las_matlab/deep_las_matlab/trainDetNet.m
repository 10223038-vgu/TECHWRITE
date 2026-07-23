function detNet = trainDetNet(M, K, hiddenSize, vSize, numIters, batchSize, learnRate)
% TRAINDETNET Train a DetNet-style unfolded detection network
% (Samuel et al., "Learning to Detect", refs [35]/[39] in the paper),
% generalized from the original BPSK formulation to M-PAM by
% normalizing the PAM grid to [-1,1] before applying the projection
% nonlinearity, and rescaling back at inference.
%
% *** THIS IS THE MOST EXPERIMENTAL FILE IN THE PACKAGE ***
% Unlike the MLP/GRU code (which uses MATLAB's standard trainNetwork
% API), DetNet's per-layer trainable step sizes and shared/step-varying
% weight matrices don't fit the Layer-array API, so this uses a manual
% dlarray + custom-Adam training loop. That style of code is much
% harder to get exactly right without execution, and dlgradient's
% support for a nested cell-array-of-structs parameter container
% (as used here) has NOT been verified in this environment. If this
% errors out, the most likely fix is flattening `params` into named
% top-level fields (params.W1_1, params.W1_2, ... instead of
% params.W1{1}, params.W1{2}, ...) so dlgradient/adamupdate have an
% unambiguous flat container to differentiate through.
%
% Architecture per layer k (real-valued, dimension Nt2 = 2*Nt):
%   z_k = x_{k-1} - delta1_k * (H'y) + delta2_k * (H'H) * x_{k-1}
%   h_k = ReLU(W1_k [z_k; x_{k-1}; v_{k-1}] + b1_k)
%   [xRaw_k; v_k] = W2_k h_k + b2_k
%   x_k = psi_{t_k}(xRaw_k)   % piecewise-linear projection onto [-1,1]
%
% Trained on freshly-generated synthetic channel realizations each
% iteration (no fixed dataset needed -- this matches how such
% detectors are normally trained, since channel realizations are
% free to sample).
%
% Inputs:
%   M          - QAM order
%   K          - number of unfolded layers (paper-style DetNet: ~10-90;
%                start small, e.g. 10, given this is untested)
%   hiddenSize - width of each layer's hidden ReLU layer
%   vSize      - size of the auxiliary memory vector v
%   numIters   - number of training iterations (mini-batches)
%   batchSize  - samples per iteration
%   learnRate  - Adam learning rate
%
% Output: detNet - struct with fields {params, K, Nt, Nr, M, hiddenSize, vSize}

if nargin < 2 || isempty(K),          K = 10;   end
if nargin < 3 || isempty(hiddenSize), hiddenSize = 40; end
if nargin < 4 || isempty(vSize),      vSize = 8; end
if nargin < 5 || isempty(numIters),   numIters = 2000; end
if nargin < 6 || isempty(batchSize),  batchSize = 20; end
if nargin < 7 || isempty(learnRate),  learnRate = 1e-3; end

cfg = getConfig();
Nt = cfg.Nt; Nr = cfg.Nr;
Nt2 = 2*Nt;
qh = qamHelpers();
Es = qh.symEnergy(M);
scaleFactor = sqrt(M) - 1;   % normalizes PAM levels {-(K-1)..K-1} to [-1,1]

inSize = 2*Nt2 + vSize;
outSize = Nt2 + vSize;

% --- initialize trainable parameters ---
params = struct();
for k = 1:K
    params.W1{k} = dlarray(0.05*randn(hiddenSize, inSize));
    params.b1{k} = dlarray(zeros(hiddenSize, 1));
    params.W2{k} = dlarray(0.05*randn(outSize, hiddenSize));
    params.b2{k} = dlarray(zeros(outSize, 1));
    params.t{k}      = dlarray(0.5);
    params.delta1{k} = dlarray(0.1);
    params.delta2{k} = dlarray(0.1);
end

% --- manual Adam optimizer state ---
mState = initZeroState(params, K);
vState = initZeroState(params, K);
beta1 = 0.9; beta2 = 0.999; adamEps = 1e-8;

for iter = 1:numIters
    snrdB = 4 + 12*rand();   % train across a spread of SNRs, per typical DetNet training
    sigma2 = Es / 10^(snrdB/10);

    batchHr = cell(batchSize,1); batchYr = cell(batchSize,1); batchXtrue = cell(batchSize,1);
    for i = 1:batchSize
        symIdx = randi([0 M-1], Nt, 1);
        x = qh.mod(symIdx, M);
        H = genChannel(Nr, Nt);
        n = sqrt(sigma2/2)*(randn(Nr,1)+1i*randn(Nr,1));
        y = H*x + n;
        [Hr, yr] = complexToReal(H, y);
        batchHr{i} = Hr; batchYr{i} = yr;
        batchXtrue{i} = [real(x); imag(x)] / scaleFactor;   % normalized target
    end

    [loss, grads] = dlfeval(@detNetLossGrad, params, batchHr, batchYr, batchXtrue, K, Nt2, vSize);

    for k = 1:K
        [params.W1{k}, mState.W1{k}, vState.W1{k}] = adamStep(params.W1{k}, grads.W1{k}, mState.W1{k}, vState.W1{k}, iter, learnRate, beta1, beta2, adamEps);
        [params.b1{k}, mState.b1{k}, vState.b1{k}] = adamStep(params.b1{k}, grads.b1{k}, mState.b1{k}, vState.b1{k}, iter, learnRate, beta1, beta2, adamEps);
        [params.W2{k}, mState.W2{k}, vState.W2{k}] = adamStep(params.W2{k}, grads.W2{k}, mState.W2{k}, vState.W2{k}, iter, learnRate, beta1, beta2, adamEps);
        [params.b2{k}, mState.b2{k}, vState.b2{k}] = adamStep(params.b2{k}, grads.b2{k}, mState.b2{k}, vState.b2{k}, iter, learnRate, beta1, beta2, adamEps);
        [params.t{k}, mState.t{k}, vState.t{k}] = adamStep(params.t{k}, grads.t{k}, mState.t{k}, vState.t{k}, iter, learnRate, beta1, beta2, adamEps);
        [params.delta1{k}, mState.delta1{k}, vState.delta1{k}] = adamStep(params.delta1{k}, grads.delta1{k}, mState.delta1{k}, vState.delta1{k}, iter, learnRate, beta1, beta2, adamEps);
        [params.delta2{k}, mState.delta2{k}, vState.delta2{k}] = adamStep(params.delta2{k}, grads.delta2{k}, mState.delta2{k}, vState.delta2{k}, iter, learnRate, beta1, beta2, adamEps);
    end

    if mod(iter, 100) == 0
        fprintf('DetNet iter %d/%d, loss = %.4f\n', iter, numIters, double(gather(extractdata(loss))));
    end
end

detNet.params = params;
detNet.K = K; detNet.Nt = Nt; detNet.Nr = Nr; detNet.M = M;
detNet.hiddenSize = hiddenSize; detNet.vSize = vSize;
detNet.scaleFactor = scaleFactor;
end

% ==================================================================
function state = initZeroState(params, K)
state = struct();
fields = {'W1','b1','W2','b2','t','delta1','delta2'};
for f = 1:numel(fields)
    fn = fields{f};
    for k = 1:K
        state.(fn){k} = zeros(size(params.(fn){k}), 'like', extractdata(params.(fn){k}));
    end
end
end

% ==================================================================
function [param, m, v] = adamStep(param, grad, m, v, t, lr, beta1, beta2, eps_)
gradVal = extractdata(grad);
m = beta1*m + (1-beta1)*gradVal;
v = beta2*v + (1-beta2)*(gradVal.^2);
mHat = m / (1 - beta1^t);
vHat = v / (1 - beta2^t);
paramVal = extractdata(param) - lr * mHat ./ (sqrt(vHat) + eps_);
param = dlarray(paramVal);
end

% ==================================================================
function [loss, grads] = detNetLossGrad(params, batchHr, batchYr, batchXtrue, K, Nt2, vSize)
nBatch = numel(batchHr);
loss = dlarray(0);
for i = 1:nBatch
    xAll = detNetForwardSample(params, batchHr{i}, batchYr{i}, K, Nt2, vSize);
    xtrue = batchXtrue{i};
    for k = 1:K
        loss = loss + log(k+1) * mean((xAll{k} - xtrue).^2);
    end
end
loss = loss / nBatch;
grads = dlgradient(loss, params);
end

% ==================================================================
function xAll = detNetForwardSample(params, Hr, yr, K, Nt2, vSize)
HtH = Hr.' * Hr;
Hty = Hr.' * yr;
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
    x = -1 + max(xRaw+tk,0)/tk - max(xRaw-tk,0)/tk;   % psi_t projection to [-1,1]
    xAll{k} = x;
end
end
