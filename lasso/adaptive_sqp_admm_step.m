function [x_next, y_next, lambda_next, beta_next, xi_next, alpha, tau_k, hessian_state] = ...
    adaptive_sqp_admm_step(Data_batch, x, y, lambda, g_est, mu, A, beta, xi, k, L, p, hessian_state, Data_hessian_pool)
%ADAPTIVE_SQP_ADMM_STEP 娑撱儲鐗哥?圭偟骞囩拋鐑樻瀮 Algorithm 1 閻ㄥ嫪绔村▎?k 鏉╊厺鍞妴?
% 瑜版挸澧犵?圭偤鐛欑痪锔芥将娑??A*x-y=0閿涘苯鑻熼柅澶婂絿 P_k=beta*p.Py*I閵??

d = length(x);
AtA = A' * A;
normA2 = norm(A, 2)^2;
if nargin < 13 || isempty(hessian_state)
    hessian_state = struct();
end
if isfield(p, 'hessian_refresh_interval')
    hessian_refresh_interval = max(1, floor(p.hessian_refresh_interval));
else
    hessian_refresh_interval = 5;
end
if isfield(p, 'hessian_min_eig')
    hessian_min_eig = max(0, p.hessian_min_eig);
else
    hessian_min_eig = 1;
end
if isfield(p, 'hessian_safety')
    hessian_safety = max(0, p.hessian_safety);
else
    hessian_safety = 1e-10;
end

% 鐠佺儤鏋冮崗顒?绱? (10)閿涙 鐎涙劙妫舵０?
temp_y = (A * x - lambda / beta + p.Py * y) / (1 + p.Py);
threshold = mu / (beta * (1 + p.Py));
y_next = sign(temp_y) .* max(abs(temp_y) - threshold, 0);

% Refresh the stochastic curvature model periodically.  The shift below
% makes the local quadratic system positive definite without eigendecomposition.
refresh_hessian = ~isfield(hessian_state, 'H_bar') || ...
    mod(k, hessian_refresh_interval) == 0;
if refresh_hessian
    % A larger curvature-only batch stabilizes the stochastic SQP direction
    % while leaving the gradient-estimator batch unchanged.
    Data_hessian = Data_batch;
    if nargin >= 14 && ~isempty(Data_hessian_pool) && ...
            isfield(p, 'hessian_batch_size') && p.hessian_batch_size > 0
        hessian_batch_size = min(size(Data_hessian_pool, 1), ...
            max(1, floor(p.hessian_batch_size)));
        hessian_indices = randperm(size(Data_hessian_pool, 1), hessian_batch_size);
        Data_hessian = Data_hessian_pool(hessian_indices, :);
    end
    H_bar = hessian_GN_twoblock(Data_hessian, x, 0, []);
    X_batch = Data_hessian(:, 1:end-1);
    batch_size_hessian = size(X_batch, 1);
    negative_curvature_bound = sum(X_batch(:).^2) / ...
        (6 * sqrt(3) * batch_size_hessian);
    tau_k = hessian_min_eig + negative_curvature_bound + hessian_safety;
    hessian_state.H_bar = H_bar;
    hessian_state.tau_k = tau_k;
    hessian_state.has_factor = false;
else
    H_bar = hessian_state.H_bar;
    tau_k = hessian_state.tau_k;
end
H_k = H_bar + tau_k * eye(d);

% 鐠佺儤鏋冮崗顒?绱? (13)
residual = A * x - y_next;
linear_term = g_est - A' * lambda + beta * A' * residual;
can_reuse_factor = ~refresh_hessian && ...
    isfield(hessian_state, 'has_factor') && hessian_state.has_factor && ...
    isfield(hessian_state, 'factor_beta') && isequal(beta, hessian_state.factor_beta);
if can_reuse_factor
    d_x = -(hessian_state.R \ (hessian_state.R' \ linear_term));
else
    H_cal = H_k + beta * AtA;
    [R, chol_flag] = chol(H_cal);
    if chol_flag == 0
        d_x = -(R \ (R' \ linear_term));
        hessian_state.R = R;
        hessian_state.factor_beta = beta;
        hessian_state.has_factor = true;
    else
        d_x = -H_cal \ linear_term;
        hessian_state.has_factor = false;
    end
end

% 鐠佺儤鏋冮崗顒?绱? (19)閵??22)
A_d = A * d_x;
epsilon_k = norm(residual, 2)^2 - norm(residual + A_d, 2)^2;
vartheta_k = 1 / (k + 1);
dHd = d_x' * H_k * d_x;

if norm(residual, 2) > vartheta_k * min(1, norm(A_d, 2))
    denominator = epsilon_k - p.zeta2 * norm(A_d, 2)^2;
    if denominator ~= 0
        denominator_dagger = 1 / denominator;
    else
        denominator_dagger = 0;
    end
    beta_trial = 2 * denominator_dagger * ...
        ((g_est - A' * lambda)' * d_x + p.zeta1 * dHd);
else
    beta_trial = 0;
end

% Penalty-update allowance: an optional linear warm start followed by a
% summable tail.  This keeps beta changes large only in the early phase.
if isfield(p, 'gamma_linear_iters')
    gamma_linear_iters = max(0, floor(p.gamma_linear_iters));
    gamma_linear_drop = max(0, p.gamma_linear_drop);
    if k <= gamma_linear_iters
        gamma_k = p.gamma_max - gamma_linear_drop * k;
    else
        gamma_transition = p.gamma_max - ...
            gamma_linear_drop * gamma_linear_iters;
        gamma_decay_index = k - gamma_linear_iters;
        if isfield(p, 'gamma_tail_ratio')
            if p.gamma_tail_ratio <= 0 || p.gamma_tail_ratio >= 1
                error('gamma_tail_ratio must lie strictly in (0, 1).');
            end
            gamma_tail_acceleration = max(0, p.gamma_tail_acceleration);
            gamma_k = gamma_transition * p.gamma_tail_ratio^gamma_decay_index * ...
                exp(-gamma_tail_acceleration * gamma_decay_index * ...
                (gamma_decay_index - 1) / 2);
        else
            gamma_k = gamma_transition / ...
                (1 + gamma_decay_index / p.gamma_scale)^p.gamma_power;
        end
    end
    if ~isfinite(gamma_k)
        error('gamma_k must be finite.');
    end
    gamma_k = max(gamma_k, realmin);
else
    if isfield(p, 'gamma_warmup')
        gamma_warmup = max(0, floor(p.gamma_warmup));
    else
        gamma_warmup = 0;
    end
    gamma_decay_index = max(0, k - gamma_warmup);
    gamma_k = p.gamma_max / ...
        (1 + gamma_decay_index / p.gamma_scale)^p.gamma_power;
end
curvature_term = (g_est - A' * lambda)' * d_x + dHd;
hat_vartheta_k = (1 - p.zeta1) / (k + 1);
if curvature_term <= hat_vartheta_k * min(1, dHd)
    beta_next = max(min((1 + gamma_k) * beta, beta_trial), beta);
elseif k < p.K0
    beta_next = max(beta / (1 + gamma_k), beta_trial);
else
    beta_next = beta;
end

% 鐠佺儤鏋冮崗顒?绱? (18)閵??38c)閵??39)
delta_q = -(g_est - A' * lambda)' * d_x + (beta_next / 2) * epsilon_k;
d_norm_sq = norm(d_x, 2)^2;
if d_norm_sq == 0
    xi_bar = Inf;
else
    xi_bar = delta_q / d_norm_sq;
end

if xi <= xi_bar
    xi_next = xi;
else
    xi_next = min((1 - p.epsilon_xi) * xi, xi_bar);
end

% 鐠佺儤鏋冮崗顒?绱? (38b)閵??40)閵??41)
alpha_bar = min(2 * (1 - p.delta) * xi_bar / (L + normA2 * beta_next), 1);
alpha_tilde = min(2 * (1 - p.delta) * xi_next / (L + normA2 * beta_next), 1);
alpha = max(alpha_tilde, min(alpha_tilde + p.varrho, alpha_bar));

% 鐠佺儤鏋冮崗顒?绱? (15)閵??45)閿涙艾顕崑鑸垫纯閺傞濞囬悽?beta_k閿涘矁?宀勬姜 beta_{k+1}
x_next = x + alpha * d_x;
lambda_next = lambda - p.s * beta * (A * x_next - y_next);

if beta_next <= 0 || ~isfinite(beta_next)
    error('beta_{k+1} must be finite and positive.');
end
if ~all(isfinite([x_next; y_next; lambda_next; xi_next; alpha; tau_k]))
    error('Algorithm 1 update produced NaN or Inf.');
end
end

