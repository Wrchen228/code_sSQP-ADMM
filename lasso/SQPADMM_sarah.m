function [test_loss_history, time_history, residual_norm, train_obj_history, x_final, x_random, ...
    beta_history, alpha_history, iteration_history, ...
    early_beta_history, early_alpha_history, early_iteration_history] = ...
    SQPADMM_sarah(Data_train, Data_test, x, y, mu, A, max_time, batch_size, p)
%SQPADMM_SARAH 论文 Algorithm 1 + SARAH 梯度估计器。

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
previous_x = x;
previous_g_est = gradient_twoblock(Data_train, x);

while elapsed_solver + toc(segment_timer) < max_time
    % The curvature model is always formed from a stochastic mini-batch.
    % A full-gradient refresh corrects only the SARAH estimator and must
    % not trigger a full-data Hessian/Gauss--Newton computation.
    idx_batch = randperm(n, min(batch_size, n));
    Data_batch = Data_train(idx_batch, :);
    use_full_refresh = (k == 0) || (rand < p.refresh_prob);
    if use_full_refresh
        g_est = gradient_twoblock(Data_train, x);
    else
        g_est = gradient_twoblock(Data_batch, x) ...
            - gradient_twoblock(Data_batch, previous_x) ...
            + previous_g_est;
    end

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
    previous_x = x;
    previous_g_est = g_est;
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
