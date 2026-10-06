function normA = estimate_operator_norm(Afun, Atfun, sz, maxIter)
%ESTIMATE_OPERATOR_NORM Power iteration for linear operator norm.
if nargin < 4
    maxIter = 30;
end
z = randn(sz);
z = z / norm(z(:));
for k = 1:maxIter
    z = Atfun(Afun(z));
    nz = norm(z(:));
    if nz == 0
        normA = 0;
        return;
    end
    z = z / nz;
end
Az = Afun(z);
normA = norm(Az(:)) / max(norm(z(:)), eps);
end
