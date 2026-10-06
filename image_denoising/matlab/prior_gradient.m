function grad = prior_gradient(x, priorType, delta, batchMask)
%PRIOR_GRADIENT Gradient of sum q(|grad x|) under periodic boundary.
[dx, dy] = gradient_periodic(x);
mag = sqrt(dx.^2 + dy.^2);
[~, coeff] = image_prior(mag, priorType, delta);
if nargin >= 4 && ~isempty(batchMask)
    coeff = coeff .* batchMask;
end
px = coeff .* dx;
py = coeff .* dy;
grad = -divergence_periodic(px, py);
end
