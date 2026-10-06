function p = Set_adult_Params(p)
%SET_ADULT_PARAMS Parameters calibrated for the Adult data set.
if nargin < 1
    p = struct();
end

p.K0 = 2000;
p.s = 0.8;
p.delta = 0.8;
p.varrho = 0.6;
p.epsilon_xi = 0.1;
p.zeta1 = 0.9;
p.zeta2 = 0.4;
% Adult residual-oriented tuning.  Keep s, delta and varrho fixed above.
% A larger mini-batch stabilizes SAGA; beta/xi improve the common
% feasibility trade-off for the three Algorithm 1 gradient estimators.
p.beta = 0.1;
p.xi0 = 0.1;
p.Py = 0.05;
p.refresh_prob = 1 / 100;
p.gamma_max = 0.12;
p.gamma_warmup = 0;
p.gamma_scale = 6;
p.gamma_power = 2;
p.record_interval = 0.5;
p.parameter_trace_max_iters = 5000;
p.parameter_trace_change_tol = 1e-5;
p.parameter_trace_tail = 5;
p.hessian_refresh_interval = 5;
p.hessian_min_eig = 1;
p.hessian_safety = 1e-10;

p.max_time = 50;
p.num_inner_iters = round(1 / p.refresh_prob);
p.batch_size = 64;
end
