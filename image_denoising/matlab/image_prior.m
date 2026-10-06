function [value, coeff] = image_prior(mag, type, delta)
%IMAGE_PRIOR Prior value and gradient coefficient for isotropic gradients.
switch lower(type)
    case 'huber'
        value = 0.5 * mag.^2;
        coeff = ones(size(mag));
        idx = mag > delta;
        value(idx) = mag(idx) - 0.5 * delta;
        coeff(idx) = 1 ./ max(mag(idx), eps);
    case 'log1p'
        % q(t) = |t| - delta * log(delta + |t|)
        % This is equivalent to |t| - delta * log(1 + |t|/delta) up to a constant.
        value = mag - delta * log(delta + mag);
        coeff = 1 ./ (delta + mag);
    case 'logratio'
        % q(t) = |t| - delta * log(1 + |t|/delta)
        value = mag - delta * log1p(mag ./ max(delta, eps));
        coeff = 1 ./ (delta + mag);
    case 'logquad'
        % q(t) = log(1 + t^2 / delta^2), q'(t) = 2t / (delta^2 + t^2)
        delta2 = max(delta, eps)^2;
        value = log1p(mag.^2 ./ delta2);
        coeff = 2 ./ (delta2 + mag.^2);
    otherwise
        error('Unknown prior type: %s', type);
end
end
