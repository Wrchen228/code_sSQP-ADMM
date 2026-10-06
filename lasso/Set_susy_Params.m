function p = Set_susy_Params(p)
%SET_SUSY_PARAMS Parameters calibrated for the SUSY data set.
if nargin < 1
    p = struct();
end

p.K0 = 5000;
% A moderate dual step gives the best feasibility-stability trade-off on SUSY.
p.s = 0.8;
p.delta = 0.85;      % Damped SQP step for lower residual variability.
p.varrho = 0.80;
p.epsilon_xi = 0.1;
p.zeta1 = 0.9;
p.zeta2 = 0.4;
p.beta = 0.1;
p.xi0 = 0.10;
p.Py = 0.10;         % Stabilize the auxiliary-variable update.
p.batch_size = 512;
p.refresh_prob = 1 / 200;
p.num_inner_iters = round(1 / p.refresh_prob);

p.gamma_max = 0.10;
p.gamma_warmup = 0;
p.gamma_scale = 8;
p.gamma_power = 2;
p.parameter_trace_max_iters = 50000;
p.parameter_trace_change_tol = 1e-4;
p.parameter_stability_window = 50;


p.hessian_refresh_interval = 100;
p.hessian_min_eig = 1;
p.hessian_safety = 1e-10;

p.record_interval = 0.5;
p.max_time = 50;
end
