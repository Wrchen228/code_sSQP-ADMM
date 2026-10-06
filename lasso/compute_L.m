function L_result = compute_L(Data)
%COMPUTE_L 计算非凸 sigmoid margin loss 的紧致全局 Lipschitz 上界。
% 对单样本损失 phi(t)=1/(1+exp(t))，有
% max_t |phi''(t)| = 1/(6*sqrt(3))。
% 因而 L <= lambda_max(X'*X)/(6*sqrt(3)*n)，
% 该谱上界比 (1/(4*n))*sum_i ||a_i||^2 更精确。

X = Data(:, 1:end-1);
n = size(X, 1);
if n == 0
    error('compute_L:EmptyData', '数据集不能为空。');
end

% X'*X 的维度仅为 d×d；对当前数据集比直接计算 norm(X,2) 更高效。
gram_matrix = X' * X;
gram_matrix = full((gram_matrix + gram_matrix') / 2);
lambda_max = max(eig(gram_matrix));
lambda_max = max(real(lambda_max), 0);

L_result = lambda_max / (6 * sqrt(3) * n);
end

