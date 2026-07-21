function xhat0 = initEstimate(y, H, sigma2, M, type)
% INITESTIMATE Linear equalizer used to seed the LAS search
% (Eqs. 5-6). type = 'zf' or 'mmse'.
qh = qamHelpers();
Nt = size(H,2);

switch lower(type)
    case 'zf'
        W = pinv(H);                 % (H'H)^-1 H'
    case 'mmse'
        Es = qh.symEnergy(M);
        W = (H'*H + (sigma2/Es)*eye(Nt)) \ H';
    otherwise
        error('initEstimate:badType', 'type must be ''zf'' or ''mmse''.');
end

xEq = W * y;
xhat0 = qh.hardDecision(xEq, M);
end
