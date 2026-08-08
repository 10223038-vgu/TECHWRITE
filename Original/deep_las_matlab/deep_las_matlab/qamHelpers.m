function s = qamHelpers()
% QAMHELPERS Returns a struct of function handles for modulation,
% hard-decision, and PAM-grid utilities matching the constellation
% A = {-(sqrt(M)-1), ..., -3, -1, 1, 3, ..., sqrt(M)-1} used in the
% paper (Section II, right after Eq. 2).
%
% Usage:
%   qh = qamHelpers();
%   x  = qh.mod(symIdx, M);        % integer symbol index -> complex symbol
%   sIdx = qh.demod(x, M);         % complex symbol -> nearest integer index
%   lvl = qh.pamLevels(M);         % PAM levels for one real dimension
%   xq  = qh.hardDecision(v, M);   % project arbitrary complex v onto grid
%   Es  = qh.symEnergy(M);         % average symbol energy (unnormalized grid)

s.mod          = @qamModRaw;
s.demod        = @qamDemodRaw;
s.pamLevels    = @pamLevels;
s.hardDecision = @hardDecisionQAM;
s.symEnergy    = @(M) (2/3)*(M-1);   % standard square-QAM average energy

end

% ---------------------------------------------------------------
function x = qamModRaw(symIdx, M)
% Unnormalized square-QAM modulation, Gray-coded, matching the
% integer PAM grid {-(sqrt(M)-1),...,-1,1,...,sqrt(M)-1}.
x = qammod(symIdx, M, 'UnitAveragePower', false);
end

function idx = qamDemodRaw(x, M)
idx = qamdemod(x, M, 'UnitAveragePower', false);
end

function lvl = pamLevels(M)
K = sqrt(M);
lvl = -(K-1):2:(K-1);
end

function xq = hardDecisionQAM(v, M)
% Project each entry of complex vector v onto the nearest point of
% the square-QAM PAM grid (real and imaginary parts independently),
% used as the Q(.) operator in Eqs. 5-6.
lvl = pamLevels(M);
re = clampToGrid(real(v), lvl);
im = clampToGrid(imag(v), lvl);
xq = re + 1i*im;
end

function out = clampToGrid(vals, lvl)
out = zeros(size(vals));
for k = 1:numel(vals)
    [~, i] = min(abs(lvl - vals(k)));
    out(k) = lvl(i);
end
end
