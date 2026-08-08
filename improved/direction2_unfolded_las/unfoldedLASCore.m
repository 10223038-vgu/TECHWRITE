function [xr_final, z_final] = unfoldedLASCore(yr, Hr, xr0, candSets, alphas)
% UNFOLDEDLASCORE Differentiable, FIXED-DEPTH unfolding of
% lasSearchCore.m's coordinate-descent local search, per THEORY_UNFOLDING.md
% "Option B": the discrete per-dimension candidate selection (which PAM
% level to move to) is KEPT EXACTLY as in the classical algorithm and is
% NOT given a gradient (there is no well-defined gradient through an
% argmin over a discrete, data-dependent candidate set) -- but the
% AMOUNT of that move actually applied is scaled by a learnable per-layer
% step size alpha_k, and THAT scaling is fully differentiable. Gradients
% w.r.t. alphas therefore flow correctly through every layer via the
% additive state-update arithmetic, even though no gradient flows through
% the "which candidate got picked" decision itself.
%
% This differs from lasSearchCore.m in two structural ways, both
% necessary for unfolding:
%   1. FIXED depth: exactly numel(alphas) sweeps are run, no
%      "break if nothing improved" early stop (a differentiable
%      computational graph needs a fixed, static structure).
%   2. Each accepted move is scaled by alpha_k instead of being applied
%      at full strength (alpha_k = 1 for every layer would reproduce
%      the classical algorithm's per-layer behavior exactly).
%
% Inputs:
%   yr, Hr    - real-valued received vector / channel matrix (PLAIN
%               numeric, not dlarray -- these are data, not learnable)
%   xr0       - initial real-valued vector (2Nt x 1), PLAIN numeric or
%               dlarray (becomes dlarray automatically once combined
%               with the dlarray alphas below)
%   candSets  - 1x(2Nt) cell array of candidate PAM levels per dimension
%               (same convention as lasSearchCore.m)
%   alphas    - K x 1 dlarray of LEARNABLE per-layer step sizes (this is
%               what gets trained; K = number of unfolded layers)
%
% Outputs:
%   xr_final, z_final - final state after K unfolded layers (dlarray if
%   alphas is dlarray, enabling dlgradient w.r.t. alphas downstream)

Q = Hr' * Hr;
xr = xr0;
z  = Hr' * (yr - Hr*xr);
Nt2 = numel(xr0);
K = numel(alphas);

for k = 1:K
    alpha_k = alphas(k);
    for n = 1:Nt2
        qnn_val = extractIfDl(Q(n,n));
        zn_val  = extractIfDl(z(n));
        xrn_val = extractIfDl(xr(n));
        levels = candSets{n};

        % --- Discrete candidate selection: computed on DETACHED numeric
        % values. This is the classical algorithm's exact decision rule,
        % unchanged -- deliberately given NO gradient (see header). ---
        bestCost = 0; bestLambda = 0;
        for c = levels
            lambda = c - xrn_val;
            if lambda == 0
                continue;
            end
            dCost = qnn_val*lambda^2 - 2*lambda*zn_val;
            if dCost < bestCost
                bestCost = dCost;
                bestLambda = lambda;
            end
        end

        if bestLambda ~= 0
            % --- Differentiable application of the chosen move, scaled
            % by the learnable alpha_k. bestLambda is a plain numeric
            % constant here (detached above); alpha_k is the dlarray
            % learnable parameter, so this line is exactly where
            % gradients w.r.t. alpha_k (and the chain back through
            % earlier layers' xr/z) enter the computational graph.
            %
            % NOTE: uses an additive one-hot update (xr + step*e_n)
            % rather than indexed assignment (xr(n) = ...) -- dlarray
            % does not reliably support indexed assignment under
            % automatic differentiation, while addition and scalar
            % multiplication are fully supported and dlgradient-safe. ---
            step = alpha_k * bestLambda;
            e_n = zeros(Nt2, 1);
            e_n(n) = 1;
            xr = xr + step * e_n;
            z = z - step * Q(:,n);
        end
    end
end

xr_final = xr;
z_final = z;
end

function v = extractIfDl(v)
if isa(v, 'dlarray')
    v = extractdata(v);
end
end
