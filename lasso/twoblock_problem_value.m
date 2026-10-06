function loss_result = twoblock_problem_value(Data, x0, y0, mu)
% 计算两分块问题的目标函数值
% min_x,y f(x) + theta(y), s.t. A*x - y = 0
% 其中 f(x) = (1/n)*sum_i 1/(1+exp(b_i*a_i'*x))，为非凸 sigmoid margin loss
%      theta(y) = mu * ||y||_1
%
% 输入：
%   Data : n×(d+1) 矩阵，前 d 列为特征，最后一列为标签（±1）
%   x0   : 模型参数列向量
%   y0   : 辅助变量列向量（应与 x0 维度相同）
%   mu   : L1 正则化系数
%
% 输出：
%   loss_result : f(x) + theta(y)

%% 计算非凸 sigmoid loss 与 L1 正则项
n = size(Data,1);           % 样本数量
x_star = Data(:,1:end-1);   % 对应样本属性
y_star = Data(:,end);       % 对应样本标签
margin = (x_star * x0) .* y_star;
sample_loss = zeros(n, 1);
positive = margin >= 0;
exp_neg_margin = exp(-margin(positive));
sample_loss(positive) = exp_neg_margin ./ (1 + exp_neg_margin);
exp_margin = exp(margin(~positive));
sample_loss(~positive) = 1 ./ (1 + exp_margin);
loss = mean(sample_loss);
loss = loss+mu*norm(y0,1); % 加上正则项
loss_result = loss; 
end
