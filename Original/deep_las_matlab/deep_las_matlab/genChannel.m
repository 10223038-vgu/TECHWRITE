function H = genChannel(Nr, Nt)
% GENCHANNEL Generate an i.i.d. complex Gaussian Rayleigh fading
% Nr x Nt MIMO channel matrix, zero mean, unit variance per entry
% (Section II, Eq. 2 and surrounding text).
H = (randn(Nr, Nt) + 1i*randn(Nr, Nt)) / sqrt(2);
end
