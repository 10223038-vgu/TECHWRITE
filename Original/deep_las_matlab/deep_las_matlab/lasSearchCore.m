function [xr_hat, Fcost, nIter] = lasSearchCore(yr, Hr, xr0, candSets, maxIter)
% LASSEARCHCORE Generic single-symbol-update local search that
% iteratively decreases the likelihood cost
%   Lambda(xr) = xr' * Q * xr - 2 * yr' * Hr * xr        (Eq. 7)
% by, for each real dimension n, testing every candidate level in
% candSets{n} and moving to the best one if it strictly decreases
% the cost (Eq. 10), then repeating sweeps until no dimension
% improves (Algorithm 1) or until maxIter sweeps are reached.
%
% This is the shared engine behind:
%   - las1Hard.m   : candSets{n} = full PAM grid for every n (Algorithm 1)
%   - modLASCounter.m : candSets{n} = full grid for n != flipped dim,
%                        candSets{flippedDim} = bit-constrained subset
%                        (Algorithm 2)
%
% Inputs:
%   yr, Hr    - real-valued received vector / channel matrix
%   xr0       - initial real-valued vector (2Nt x 1)
%   candSets  - 1x(2Nt) cell array; candSets{n} = allowed levels for
%               dimension n on this call
%   maxIter   - safety cap on outer sweeps
%
% Outputs:
%   xr_hat  - final real-valued vector
%   Fcost   - final likelihood cost, Eq. 7
%   nIter   - number of accepted per-dimension updates

Q  = Hr' * Hr;
xr = xr0;
z  = Hr' * (yr - Hr*xr);     % residual correlation vector
Nt2 = numel(xr);

nIter = 0;
for outer = 1:maxIter
    improvedAny = false;
    for n = 1:Nt2
        qnn = Q(n,n);
        zn  = z(n);
        levels = candSets{n};

        bestCost = 0;              % cost delta of "no move" is 0
        bestLambda = 0;
        for c = levels
            lambda = c - xr(n);
            if lambda == 0
                continue;
            end
            dCost = qnn*lambda^2 - 2*lambda*zn;   % Eq. 10
            if dCost < bestCost
                bestCost = dCost;
                bestLambda = lambda;
            end
        end

        if bestLambda ~= 0
            xr(n) = xr(n) + bestLambda;
            z = z - bestLambda * Q(:,n);          % Eqs. 12-13
            nIter = nIter + 1;
            improvedAny = true;
        end
    end
    if ~improvedAny
        break;
    end
end

xr_hat = xr;
Fcost = xr_hat'*Q*xr_hat - 2*yr'*Hr*xr_hat;
end
