function [Hr, yr] = complexToReal(H, y)
% COMPLEXTOREAL Convert a complex Nr x Nt MIMO system y = Hx + n into
% its real-valued equivalent (2Nr x 2Nt), which is the standard
% representation used to run LAS-type neighborhood search directly
% on the real/imaginary PAM grid (consistent with the real/imag
% vectorization used for the DNN inputs in Eq. 19 of the paper).
%
%   Hr = [ Re(H)  -Im(H) ;
%          Im(H)   Re(H) ]
%   yr = [ Re(y) ; Im(y) ]
%
% A real solution vector xr = [Re(x); Im(x)] (length 2*Nt) recovers
% the complex symbol vector via x = xr(1:Nt) + 1i*xr(Nt+1:end).

Hr = [real(H), -imag(H); imag(H), real(H)];
yr = [real(y); imag(y)];
end
