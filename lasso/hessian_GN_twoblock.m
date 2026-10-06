function H = hessian_GN_twoblock(Data, x, tau, batch_size)
% 非凸 sigmoid margin loss 的真实 Hessian（可选随机小批量）+ tau*I
% 单样本损失：f_i(x) = 1/(1+exp(b_i*a_i'*x))，要求标签 b_i ∈ {-1,1}
% tau*I 用于抑制负曲率；只有 tau 足够大时才能保证所得矩阵一致正定
% 输入:
%   Data       : [B, b] 矩阵，大小为 n x (d+1)
%   x          : 当前迭代点，大小为 d x 1
%   tau        : 正则化系数 (若为了严格保证一致正定，建议在外部通过特征值或Gershgorin动态调整)
%   batch_size : (可选) 随机采样的小批量大小。如果不传或传空，则默认使用全量数据 (n)

B_matrix = Data(:, 1:end-1);   % n x d  特征矩阵 (A)
b = Data(:, end);              % n x 1  标签
n = size(B_matrix, 1);
d = size(B_matrix, 2);

% ==================== 1. 随机小批量采样 (Stochastic Sampling) ====================
if nargin < 4 || isempty(batch_size) || batch_size >= n
    % 使用全量数据
    idx = 1:n;
    current_batch_size = n;
else
    % 随机抽取 batch_size 个样本的索引
    idx = randperm(n, batch_size);
    current_batch_size = batch_size;
end

B_sub = B_matrix(idx, :);      % |B_k| x d
b_sub = b(idx);                % |B_k| x 1

% ==================== 2. 计算 Sigmoid 值并防止溢出 ====================
% 根据你的模型定义：f_i(x) = 1 / (1 + exp(b_i * a_i' * x))
% 注意：此处直接将 b_i 乘入指数中
s = (B_sub * x) .* b_sub;      % |B_k| x 1

% 分段计算，避免 exp(s) 在大正间隔上溢
f = zeros(current_batch_size, 1);
positive = s >= 0;
exp_neg_s = exp(-s(positive));
f(positive) = exp_neg_s ./ (1 + exp_neg_s);
exp_s = exp(s(~positive));
f(~positive) = 1 ./ (1 + exp_s);

% ==================== 3. 计算真实 Hessian 权重 ====================
% 一般公式：w_i = b_i^2*f_i*(1-f_i)*(1-2*f_i)
% 当前标签 b_i ∈ {-1,1}，因此 b_i^2=1，可省略该因子
% 注意：当 f_i > 0.5 时，(1 - 2*f_i) < 0，权重为负，体现非凸性
w = f .* (1 - f) .* (1 - 2 * f);  % |B_k| x 1

% ==================== 4. 组装 Hessian 矩阵 ====================
% 核心：计算 \tilde{H}_k = (1/|B_k|) * B_sub' * diag(w) * B_sub
% 使用 bsxfun 避免显式创建巨大的对角矩阵，成倍提升运行速度
H_tilde = (B_sub' * bsxfun(@times, w, B_sub)) / current_batch_size;

% 确保数值上的绝对对称性 (防止浮点数微小截断误差)
H_tilde = (H_tilde + H_tilde') / 2;

% ==================== 5. Levenberg--Marquardt 型正则化 ====================
% H_k = \tilde{H}_k + \tau * I
% 注意：tau 必须大于负曲率幅度才能保证 H_k 正定
H = H_tilde + tau * eye(d);

end
