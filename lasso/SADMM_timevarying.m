function [test_loss_history, time_history, residual_norm, train_obj_history, x_final, x_random] = ...
    SADMM_timevarying(Data_train, Data_test, x, y, mu, A, max_time, record_interval, rho)
%SADMM_TIMEVARYING Huang--Chen (2016) Algorithm 1 with eta_t=2*sqrt(t+1).
% Uses B=-I, c=0, Q=I, fixed rho, and one uniformly sampled component
% gradient per iteration.  The effective gradient step is 1/eta_t.

if nargin < 8
    record_interval = 0.5;
end
if nargin < 9
    rho = 0.1;
end

capacity = 5000;
test_loss_history = zeros(capacity, 1);
time_history = zeros(capacity, 1);
residual_norm = zeros(capacity, 1);
train_obj_history = zeros(capacity, 1);

n = size(Data_train, 1);
d = length(x);
lambda = zeros(size(A, 1), 1);
AtA = A' * A;

record_count = 1;
test_loss_history(1) = twoblock_problem_value(Data_test, x, zeros(size(x)), 0);
train_obj_history(1) = twoblock_problem_value(Data_train, x, y, mu);
residual_norm(1) = norm(A * x - y, 1);
time_history(1) = 0;
x_last = x;
y_last = y;
x_random = x;

output_stream = RandStream('mt19937ar', 'Seed', 314159);
elapsed_solver = 0;
segment_timer = tic;
next_record_time = record_interval;
t = 0;

while elapsed_solver + toc(segment_timer) < max_time
    % Article experiment: eta_t=2*sqrt(t), written safely from t=0 as
    % eta_t=2*sqrt(t+1).  Q=I gives the exact surrogate minimizer below.
    eta_t = 2 * sqrt(t + 1);

    % Algorithm 1, steps 4--6, specialized to B=-I and c=0.
    sample_id = randi(n);
    g_sample = gradient_twoblock(Data_train(sample_id, :), x);
    temp_y = A * x - lambda / rho;
    y_next = sign(temp_y) .* max(abs(temp_y) - mu / rho, 0);

    linear_system = eta_t * speye(d) + rho * AtA;
    rhs = eta_t * x - g_sample + rho * A' * (y_next + lambda / rho);
    [R, chol_flag] = chol(linear_system);
    if chol_flag == 0
        x_next = R \ (R' \ rhs);
    else
        x_next = linear_system \ rhs;
    end
    lambda_next = lambda - rho * (A * x_next - y_next);

    completed_time = elapsed_solver + toc(segment_timer);
    if completed_time > max_time
        break;
    end
    elapsed_solver = completed_time;
    x = x_next;
    y = y_next;
    lambda = lambda_next;
    x_last = x;
    y_last = y;
    t = t + 1;

    % The paper's theoretical output is a uniformly selected inner iterate.
    if rand(output_stream) < 1 / t
        x_random = x;
    end

    if elapsed_solver >= next_record_time
        record_count = record_count + 1;
        test_loss_history(record_count) = ...
            twoblock_problem_value(Data_test, x, zeros(size(x)), 0);
        train_obj_history(record_count) = ...
            twoblock_problem_value(Data_train, x, y, mu);
        residual_norm(record_count) = norm(A * x - y, 1);
        time_history(record_count) = elapsed_solver;
        next_record_time = record_interval * ...
            (floor(elapsed_solver / record_interval) + 1);
        segment_timer = tic;
    end
end

if elapsed_solver > time_history(record_count) + eps(max(1, elapsed_solver))
    record_count = record_count + 1;
    test_loss_history(record_count) = ...
        twoblock_problem_value(Data_test, x_last, zeros(size(x_last)), 0);
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
