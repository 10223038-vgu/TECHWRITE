function [xr_final, z_final] = unfoldedLASCoreSoft(yr, Hr, xr0, candSets, rawTemps)
% UNFOLDEDLASCORESOFT Option A: fully differentiable soft relaxation of
% lasSearchCore.m's coordinate-descent local search.
%
% Unlike unfoldedLASCore.m (Option B), there is NO detached/hard decision
% anywhere in this function -- every per-dimension update is a genuine
% differentiable function of the learnable temperature and everything
% upstream. Gradients flow through the candidate selection ITSELF now,
% not just through how much of a pre-selected move gets applied.
%
% Per dimension n at layer k, instead of picking the single best
% candidate (hard argmin), compute a soft-assignment probability over
% ALL candidates and move to their probability-weighted average:
%   cost(c) = qnn*(c - xr(n))^2 - 2*(c - xr(n))*zn      [for every c in levels]
%   p(c)    = softmax(-cost(c) / (qnn*T_k))              [T_k = learnable temperature]
%   xr(n)_new = sum_c p(c) * c
% As T_k -> 0, softmax sharpens toward a one-hot at the best candidate,
% recovering the classical hard decision; larger T_k blends multiple
% candidates. Costs are normalized by qnn (a natural per-dimension
% curvature/scale term) before dividing by T_k, so T_k stays roughly
% dimensionless across different channel/SNR conditions instead of
% needing to be re-tuned per noise level.
%
% z is updated using the ACTUAL net change applied (lambda_eff = new -
% old), which keeps the same fast incremental update valid regardless of
% whether that change came from a hard or soft decision -- the
% underlying linear algebra doesn't care.
%
% Inputs:
%   yr, Hr    - real-valued received vector / channel matrix (data, not
%               learnable)
%   xr0       - initial real-valued vector (2Nt x 1)
%   candSets  - 1x(2Nt) cell array of candidate PAM levels per dimension
%   rawTemps  - K x 1 dlarray, UNCONSTRAINED learnable parameters; the
%               actual temperature per layer is T_k = exp(rawTemps(k))
%               (guarantees positivity while keeping the parameter
%               itself unconstrained, standard trick for a
%               positive-only learnable scalar)
%
% Outputs: xr_final, z_final -- dlarray if rawTemps is dlarray.

Q = Hr' * Hr;
xr = xr0;
z  = Hr' * (yr - Hr*xr);
Nt2 = numel(xr0);
K = numel(rawTemps);

for k = 1:K
    T_k = exp(rawTemps(k));   % positivity via exp -- differentiable, standard
    for n = 1:Nt2
        qnn = Q(n,n);         % plain numeric (Hr/Q are data, not learnable)
        zn  = z(n);           % may be dlarray (tracks earlier layers' gradients)
        xrn_old = xr(n);
        levels = candSets{n}; % plain numeric row vector

        lambda = levels - xrn_old;               % 1 x nLevels, dlarray if xrn_old is
        costs  = qnn*(lambda.^2) - 2*lambda.*zn;  % 1 x nLevels
        logits = -costs / (qnn*T_k);
        p = softmaxStable(logits);                % 1 x nLevels, sums to 1

        xrn_new = sum(p .* levels);                % scalar, differentiable soft blend
        lambda_eff = xrn_new - xrn_old;

        e_n = zeros(Nt2, 1);
        e_n(n) = 1;
        xr = xr + lambda_eff * e_n;
        z  = z  - lambda_eff * Q(:,n);
    end
end

xr_final = xr;
z_final = z;
end

function p = softmaxStable(logits)
% Numerically stable softmax over a row vector, dlarray-safe (uses only
% max/exp/sum/subtraction/division, all fully supported for dlarray and
% dlgradient).
m = max(logits, [], 2);
e = exp(logits - m);
p = e ./ sum(e, 2);
end
