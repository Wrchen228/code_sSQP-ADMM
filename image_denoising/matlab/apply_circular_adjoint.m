function y = apply_circular_adjoint(x, kernel)
%APPLY_CIRCULAR_ADJOINT Adjoint of circular convolution via FFT.
sz = size(x);
Fk = conj(fft2(kernel, sz(1), sz(2)));
y = real(ifft2(fft2(x) .* Fk));
end
