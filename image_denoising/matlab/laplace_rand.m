function x = laplace_rand(sz, mu, scale, seed)
%LAPLACE_RAND Draw i.i.d. Laplace random variables by inverse transform.
if nargin < 2 || isempty(mu)
    mu = 0;
end
if nargin < 3 || isempty(scale)
    scale = 1;
end
if nargin >= 4 && ~isempty(seed)
    rng(seed, 'twister');
end

u = rand(sz) - 0.5;
x = mu - scale .* sign(u) .* log1p(-2 * abs(u));
end
