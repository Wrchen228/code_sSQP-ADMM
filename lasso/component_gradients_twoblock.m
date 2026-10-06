function gradients = component_gradients_twoblock(Data, x)
%COMPONENT_GRADIENTS_TWOBLOCK Per-sample gradients of sigmoid margin loss.
% The j-th column is grad f_j(x); no averaging is applied.

A = Data(:, 1:end-1);
b = Data(:, end);
s = (A * x) .* b;

loss_value = zeros(size(s));
positive = s >= 0;
exp_neg_s = exp(-s(positive));
loss_value(positive) = exp_neg_s ./ (1 + exp_neg_s);
exp_s = exp(s(~positive));
loss_value(~positive) = 1 ./ (1 + exp_s);

coeff = -(b .* loss_value .* (1 - loss_value));
gradients = bsxfun(@times, A', coeff');
end
