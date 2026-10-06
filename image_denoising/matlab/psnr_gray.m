function value = psnr_gray(x, xref)
%PSNR_GRAY Peak signal-to-noise ratio for grayscale images in [0, 1].
err = x - xref;
mse = mean(err(:) .^ 2);
if mse <= 0
    value = inf;
else
    value = 10 * log10(1 / mse);
end
end
