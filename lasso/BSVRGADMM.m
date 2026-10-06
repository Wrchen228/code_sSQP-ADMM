function [test_loss_history, time_history, residual_norm, train_obj_history, x_final] = ...
    BSVRGADMM(Data_train, Data_test, x, y, mu, A, max_time, ...
    batch_size, epoch_length, record_interval, beta0)
%BSVRGADMM Bi et al. (2021) Algorithm 1 + Definition 2.5 SVRG.
% Uses B=-I, c=0, Q=tau*I and the scaled dual u_scaled=u/beta.

if nargin < 10
    record_interval = 0.5;
end
if nargin < 11
    beta0 = 0.1;
end

capacity = 5000;
test_loss_history = zeros(capacity, 1);
time_history = zeros(capacity, 1);
residual_norm = zeros(capacity, 1);
train_obj_history = zeros(capacity, 1);

n = size(Data_train, 1);
beta = beta0;
sigma = 1;
tau = 1;
u_scaled = zeros(size(A, 1), 1);

record_count = 1;
test_loss_history(1) = twoblock_problem_value(Data_test, x, zeros(size(y)), 0);
train_obj_history(1) = twoblock_problem_value(Data_train, x, y, mu);
residual_norm(1) = norm(A * x - y, 1);
time_history(1) = 0;
x_last = x;
y_last = y;

elapsed_solver = 0;
segment_timer = tic;
next_record_time = record_interval;

while elapsed_solver + toc(segment_timer) < max_time
    % Definition 2.5: phi^s is refreshed every m=epoch_length steps.
    snapshot = x;
    full_grad_snapshot = gradient_twoblock(Data_train, snapshot);
    time_exceeded = false;

    for inner_step = 1:epoch_length
        if elapsed_solver + toc(segment_timer) >= max_time
            time_exceeded = true;
            break;
        end

        x_current = x;
        idx_batch = randperm(n, min(batch_size, n));
        Data_batch = Data_train(idx_batch, :);
        v = gradient_twoblock(Data_batch, x_current) ...
            - gradient_twoblock(Data_batch, snapshot) + full_grad_snapshot;

        % Algorithm 1, specialized to z=y, B=-I and c=0.
        temp_y = A * x_current + u_scaled;
        y_next = sign(temp_y) .* max(abs(temp_y) - mu / beta, 0);
        x_next = x_current - (v + beta * A' * ...
            (A * x_current + u_scaled - y_next)) / tau;
        u_next = u_scaled + sigma * (A * x_next - y_next);

        completed_time = elapsed_solver + toc(segment_timer);
        if completed_time > max_time
            time_exceeded = true;
            break;
        end

        elapsed_solver = completed_time;
        x = x_next;
        y = y_next;
        u_scaled = u_next;
        x_last = x;
        y_last = y;

        if elapsed_solver >= next_record_time
            record_count = record_count + 1;
            test_loss_history(record_count) = ...
                twoblock_problem_value(Data_test, x, zeros(size(y)), 0);
            train_obj_history(record_count) = ...
                twoblock_problem_value(Data_train, x, y, mu);
            residual_norm(record_count) = norm(A * x - y, 1);
            time_history(record_count) = elapsed_solver;
            next_record_time = record_interval * ...
                (floor(elapsed_solver / record_interval) + 1);
            segment_timer = tic;
        end
    end

    if time_exceeded
        break;
    end
end

if elapsed_solver > time_history(record_count) + eps(max(1, elapsed_solver))
    record_count = record_count + 1;
    test_loss_history(record_count) = ...
        twoblock_problem_value(Data_test, x_last, zeros(size(y_last)), 0);
    train_obj_history(record_count) = ...
        twoblock_problem_value(Data_train, x_last, y_last, mu);
    residual_norm(record_count) = norm(A * x_last - y_last, 1);
    time_history(record_count) = elapsed_solver;
end

test_loss_history = test_loss_history(1:record_count);
time_history = time_history(1:record_count);
residual_norm = residual_norm(1:record_count);
train_obj_history = train_obj_history(1:record_count);
x_final = x_last;
end
