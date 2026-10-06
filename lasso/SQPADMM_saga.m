function [test_loss_history, time_history, residual_norm, train_obj_history, x_final, x_random, ...
    beta_history, alpha_history, iteration_history, ...
    early_beta_history, early_alpha_history, early_iteration_history] = ...
    SQPADMM_saga(Data_train, Data_test, x, y, mu, A, max_time, batch_size, p)
%SQPADMM_SAGA 论文 Algorithm 1 + SAGA 梯度估计器。

capacity = 5000;
test_loss_history = zeros(capacity, 1);
time_history = zeros(capacity, 1);
residual_norm = zeros(capacity, 1);
train_obj_history = zeros(capacity, 1);
parameter_capacity = 10000;
beta_history = zeros(parameter_capacity, 1);
alpha_history = nan(parameter_capacity, 1);
iteration_history = zeros(parameter_capacity, 1);
if isfield(p, 'outer_record_interval')
    outer_record_interval = max(1, floor(p.outer_record_interval));
else
    outer_record_interval = max(1, floor(p.num_inner_iters));
end

n = size(Data_train, 1);
d = length(x);
lambda = zeros(size(A, 1), 1);
beta = p.beta;
xi = p.xi0;
L = compute_L(Data_train);

record_count = 1;
test_loss_history(1) = twoblock_problem_value(Data_test, x, zeros(size(x)), 0);
train_obj_history(1) = twoblock_problem_value(Data_train, x, y, mu);
residual_norm(1) = norm(A * x - y, 1);
time_history(1) = 0;
beta_history(1) = beta;
iteration_history(1) = 0;
parameter_count = 1;
early_limit = 2000;
if isfield(p, 'parameter_trace_max_iters')
    early_limit = p.parameter_trace_max_iters;
end
early_beta_history = nan(early_limit + 1, 1);
early_alpha_history = nan(early_limit + 1, 1);
early_iteration_history = (0:early_limit)';
early_beta_history(1) = beta;

x_final = x;
x_random = x;
output_stream = RandStream('mt19937ar', 'Seed', 271828);

elapsed_solver = 0;
segment_timer = tic;
next_record_time = p.record_interval;
k = 0;
hessian_state = struct();

% 注 4.1：所有记忆点初始化为 x^0
grad_table = zeros(d, n);
grad_sum = zeros(d, 1);
if isfield(p, 'saga_init_chunk_size')
    init_chunk_size = max(1, floor(p.saga_init_chunk_size));
else
    init_chunk_size = 100000;
end
for chunk_start = 1:init_chunk_size:n
    chunk_end = min(chunk_start + init_chunk_size - 1, n);
    chunk_indices = chunk_start:chunk_end;
    chunk_gradients = component_gradients_twoblock( ...
        Data_train(chunk_indices, :), x);
    grad_table(:, chunk_indices) = chunk_gradients;
    grad_sum = grad_sum + sum(chunk_gradients, 2);
end
grad_avg = grad_sum / n;

while elapsed_solver + toc(segment_timer) < max_time
    idx_batch = randperm(n, min(batch_size, n));
    Data_batch = Data_train(idx_batch, :);

    grad_avg_old = grad_avg;
    diff_sum = zeros(d, 1);
    for batch_position = 1:numel(idx_batch)
        sample_id = idx_batch(batch_position);
        g_new = gradient_twoblock(Data_train(sample_id, :), x);
        g_old = grad_table(:, sample_id);
        diff_sum = diff_sum + g_new - g_old;
        grad_table(:, sample_id) = g_new;
        grad_avg = grad_avg + (g_new - g_old) / n;
    end
    g_est = diff_sum / numel(idx_batch) + grad_avg_old;

    if elapsed_solver + toc(segment_timer) >= max_time
        break;
    end

    [x_next, y_next, lambda_next, beta_next, xi_next, alpha, ~, hessian_state] = ...
        adaptive_sqp_admm_step(Data_batch, x, y, lambda, g_est, ...
        mu, A, beta, xi, k, L, p, hessian_state, Data_train);

    completed_time = elapsed_solver + toc(segment_timer);
    if completed_time > max_time
        break;
    end

    elapsed_solver = completed_time;
    x = x_next;
    y = y_next;
    lambda = lambda_next;
    beta = beta_next;
    xi = xi_next;
    k = k + 1;
    x_final = x;

    if k <= early_limit
        early_beta_history(k + 1) = beta;
        early_alpha_history(k + 1) = alpha;
        if isnan(early_alpha_history(1))
            early_alpha_history(1) = alpha;
        end
    end

    if mod(k, outer_record_interval) == 0
        parameter_count = parameter_count + 1;
        if parameter_count > numel(beta_history)
            new_capacity = 2 * numel(beta_history);
            beta_history(new_capacity, 1) = 0;
            alpha_history(new_capacity, 1) = NaN;
            iteration_history(new_capacity, 1) = 0;
        end
        iteration_history(parameter_count) = k / outer_record_interval;
        beta_history(parameter_count) = beta;
        alpha_history(parameter_count) = alpha;
        if isnan(alpha_history(1))
            alpha_history(1) = alpha;
        end
    end

    if rand(output_stream) < 1 / (k + 1)
        x_random = x;
    end

    if elapsed_solver >= next_record_time
        record_count = record_count + 1;
        test_loss_history(record_count) = twoblock_problem_value(Data_test, x, zeros(size(x)), 0);
        train_obj_history(record_count) = twoblock_problem_value(Data_train, x, y, mu);
        residual_norm(record_count) = norm(A * x - y, 1);
        time_history(record_count) = elapsed_solver;
        next_record_time = p.record_interval * ...
            (floor(elapsed_solver / p.record_interval) + 1);
    end
    segment_timer = tic;
end

% 无论是否正好到达记录时刻，都保存最后一个已完成迭代点
if elapsed_solver > time_history(record_count) + eps(max(1, elapsed_solver))
    record_count = record_count + 1;
    test_loss_history(record_count) = twoblock_problem_value(Data_test, x, zeros(size(x)), 0);
    train_obj_history(record_count) = twoblock_problem_value(Data_train, x, y, mu);
    residual_norm(record_count) = norm(A * x - y, 1);
    time_history(record_count) = elapsed_solver;
end

test_loss_history = test_loss_history(1:record_count);
train_obj_history = train_obj_history(1:record_count);
residual_norm = residual_norm(1:record_count);
time_history = time_history(1:record_count);
beta_history = beta_history(1:parameter_count);
alpha_history = alpha_history(1:parameter_count);
iteration_history = iteration_history(1:parameter_count);
early_count = min(k, early_limit) + 1;
early_beta_history = early_beta_history(1:early_count);
early_alpha_history = early_alpha_history(1:early_count);
early_iteration_history = early_iteration_history(1:early_count);
end
