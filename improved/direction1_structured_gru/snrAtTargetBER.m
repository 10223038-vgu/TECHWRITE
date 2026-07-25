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

if targetBER < min(berCurve(valid)) || targetBER > max(berCurve(valid))
    warning('snrAtTargetBER:extrapolating', ...
        'Target BER %.1e is outside the measured range [%.1e, %.1e] -- extrapolating.', ...
        targetBER, min(berCurve(valid)), max(berCurve(valid)));
end

snrAtTarget = interp1(logBER, snrValid, log10(targetBER), 'linear', 'extrap');
end
