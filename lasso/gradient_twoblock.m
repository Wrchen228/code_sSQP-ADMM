function result =  gradient_twoblock(Data, x)
%% 非凸 sigmoid margin loss 的梯度
% 单样本损失：q_i(x) = 1/(1+exp(b_i*a_i'*x))，其中 b_i ∈ {-1,1}
% 梯度：grad f(x) = -(1/n)*sum_i b_i*q_i(x)*(1-q_i(x))*a_i
A = Data(:,1:end-1);   % n x d
b = Data(:,end);       % n x 1
n = size(A,1);

s = (A * x) .* b;            % n x 1, s_i = b_i*(a_i'*x)
f = zeros(n, 1);
positive = s >= 0;
exp_neg_s = exp(-s(positive));
f(positive) = exp_neg_s ./ (1 + exp_neg_s);
exp_s = exp(s(~positive));
f(~positive) = 1 ./ (1 + exp_s);

% 每个样本的梯度系数：-b_i*f_i*(1-f_i)
coeff = - (b .* (f .* (1 - f)));   % n x 1
g = (A' * coeff) / n;  % d x 1
result = g;
end
