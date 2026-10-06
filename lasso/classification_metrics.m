function [error_rate, balanced_accuracy] = classification_metrics(Data, x)
%CLASSIFICATION_METRICS 计算二分类错误率与 balanced accuracy。

features = Data(:, 1:end-1);
labels = Data(:, end);
scores = features * x;

predictions = ones(size(scores));
predictions(scores < 0) = -1;
error_rate = full(mean(predictions ~= labels));

positive = labels == 1;
negative = labels == -1;
if ~any(positive) || ~any(negative)
    error('Balanced accuracy 要求测试集中同时包含正、负样本。');
end

true_positive_rate = full(mean(predictions(positive) == 1));
true_negative_rate = full(mean(predictions(negative) == -1));
balanced_accuracy = 0.5 * (true_positive_rate + true_negative_rate);
end
