function y = apply_circular_convolution(x, kernel)
%APPLY_CIRCULAR_CONVOLUTION Circular convolution via FFT.
sz = size(x);
Fk = fft2(kernel, sz(1), sz(2));
y = real(ifft2(fft2(x) .* Fk));
end
