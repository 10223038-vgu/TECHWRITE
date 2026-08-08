function Hk_all = genMultipathChannel(Nr, Nt, N, L, tau)
% GENMULTIPATHCHANNEL Generate N correlated per-subcarrier MIMO channel
% matrices from an L-tap frequency-selective channel with an
% exponential power delay profile (THEORY.md, Section 1.3).
%
%   sigma_l^2 = exp(-l/tau) / sum_{m=0}^{L-1} exp(-m/tau),   l = 0..L-1
%   h_ij[l] ~ CN(0, sigma_l^2), independent across l, i, j
%   H_k[i,j] = sum_l h_ij[l] * exp(-1i*2*pi*k*l/N),          k = 0..N-1
%
% H_k[i,j] is computed as the N-point DFT of the L-tap vector h_ij (MATLAB's
% fft(h,N) zero-pads to length N and computes exactly this sum), so every
% subcarrier's channel is a linear combination of the SAME L taps -- this
% is what makes subcarriers correlated (THEORY.md Section 1.4) rather than
% i.i.d. as in the original per-subcarrier model.
%
% Each entry H_k[i,j] is still marginally CN(0,1) (since
% sum_l sigma_l^2 = 1 and it's a sum of independent complex Gaussians),
% matching the original i.i.d.-Rayleigh model's per-entry statistics --
% only the joint correlation ACROSS subcarriers k differs. This is
% intentional: it means components trained on the original flat/i.i.d.
% data (e.g. the MLP block) remain valid without retraining, isolating
% the GRU's sequence-ordering as the only variable under test.
%
% Inputs:
%   Nr, Nt - receive/transmit antennas
%   N      - number of subcarriers (e.g. FFT length)
%   L      - number of channel taps (L=1 recovers the original i.i.d. model)
%   tau    - power-delay-profile decay parameter (larger tau = more
%            spread-out delay profile = more frequency-selective)
%
% Output:
%   Hk_all - N x Nr x Nt complex array; Hk_all(k,:,:) is the Nr x Nt
%            channel matrix for subcarrier k (k=1..N, i.e. subcarrier
%            index k-1 in the 0-indexed formula above)

if L < 1
    error('genMultipathChannel:badL', 'L must be >= 1.');
end

l = (0:L-1);
sigma2 = exp(-l/tau);
sigma2 = sigma2 / sum(sigma2);   % normalize total tap energy to 1

Hk_all = zeros(N, Nr, Nt);
for i = 1:Nr
    for j = 1:Nt
        h = sqrt(sigma2/2) .* (randn(1,L) + 1i*randn(1,L));  % L taps
        Hvec = fft(h, N);   % N-point DFT, zero-padded -> H_k for k=0..N-1
        Hk_all(:, i, j) = Hvec(:);
    end
end
end
