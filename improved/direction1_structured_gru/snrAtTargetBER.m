function snrAtTarget = snrAtTargetBER(snrRange, berCurve, targetBER)
% SNRATTARGETBER Interpolate (in log10(BER) vs SNR space, where the
% relationship is approximately linear) the SNR needed to reach a given
% target BER on a measured BER-vs-SNR curve. Used to compute the same
% "SNR gain relative to a baseline, at BER=X" metric the paper itself
% reports (e.g. "2.55 dB gain at BER=1e-4"), so our results are directly
% comparable to the paper's headline numbers.

valid = berCurve > 0;
if sum(valid) < 2
    warning('snrAtTargetBER:tooFewPoints', 'Not enough nonzero BER points to interpolate.');
    snrAtTarget = NaN;
    return;
end

logBER = log10(berCurve(valid));
snrValid = snrRange(valid);

% interp1 requires unique points along the interpolation axis (logBER
% here, since we interpolate SNR as a function of log10(BER)). A flat or
% non-monotonic curve -- e.g. an undertrained/underperforming detector
% whose BER barely moves across the SNR sweep -- can easily produce two
% SNR points with the EXACT same measured BER, which otherwise crashes
% interp1 with "Sample points must be unique." Average the SNRs sharing
% a BER value and de-duplicate before interpolating; this doesn't change
% behavior at all on a well-behaved (strictly decreasing) curve.
[logBER_u, ~, ic] = unique(logBER, 'stable');
if numel(logBER_u) < numel(logBER)
    snrValid_u = accumarray(ic(:), snrValid(:), [], @mean);
else
    snrValid_u = snrValid;
end
[logBER_u, sortIdx] = sort(logBER_u);
snrValid_u = snrValid_u(sortIdx);

if numel(logBER_u) < 2
    warning('snrAtTargetBER:tooFewUniquePoints', ...
        'BER curve is flat (only one distinct BER value) -- cannot interpolate an SNR-at-target.');
    snrAtTarget = NaN;
    return;
end

if targetBER < min(berCurve(valid)) || targetBER > max(berCurve(valid))
    warning('snrAtTargetBER:extrapolating', ...
        'Target BER %.1e is outside the measured range [%.1e, %.1e] -- extrapolating.', ...
        targetBER, min(berCurve(valid)), max(berCurve(valid)));
end

snrAtTarget = interp1(logBER_u, snrValid_u, log10(targetBER), 'linear', 'extrap');
end
