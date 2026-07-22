function xEq = initEstimateSoft(y, H, sigma2, M, type)
% INITESTIMATESOFT Linear equalizer WITHOUT hard decision -- returns
% the raw continuous ZF/MMSE output. This is the quantity that
% should feed the Deep LAS DNN (and the Fig. 5-8 scatter plots),
% because it still carries the noise/interference information the
% network needs to estimate confidence (LLR magnitude). The
% hard-decided version in initEstimate.m collapses every noise
% realization that quantizes to the same constellation point onto a
% single value, which destroys exactly the information a soft-output
% estimator needs -- feeding hard-decided values into the MLP/GRU
% makes it structurally impossible for the network to learn anything
% beyond a near-constant per-symbol output (see README.md for the
% full explanation of why this caused the Fig. 5-8 / Fig. 9 issues).
%
% type = 'zf' or 'mmse'.

qh = qamHelpers();
Nt = size(H,2);

switch lower(type)
    case 'zf'
        W = pinv(H);
    case 'mmse'
        Es = qh.symEnergy(M);
        W = (H'*H + (sigma2/Es)*eye(Nt)) \ H';
    otherwise
        error('initEstimateSoft:badType', 'type must be ''zf'' or ''mmse''.');
end

xEq = W * y;   % NOTE: no hardDecision() call here, unlike initEstimate.m
end
