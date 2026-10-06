function p = Set_covtype_binary_scaled_Params(p)
%SET_COVTYPE_BINARY_SCALED_PARAMS Initial parameters for covtype.binary.scale.
if nargin < 1
    p = struct();
end

p.K0 = 500;
p.s = 0.8;
p.delta = 0.7;
p.varrho = 0.80;
p.epsilon_xi = 0.1;
p.zeta1 = 0.9;
p.zeta2 = 0.4;
p.beta = 0.1;
p.xi0 = 0.10;
p.Py = 0.05;
p.batch_size = 256;
p.refresh_prob = 1 / 5000;
p.num_inner_iters = round(1 / p.refresh_prob);

p.gamma_max = 0.10;
p.gamma_warmup = 0;
p.gamma_scale = 8;
p.gamma_power = 2;
p.parameter_trace_max_iters = 5000;
p.parameter_trace_change_tol = 1e-5;
p.parameter_stability_window = 50;


p.hessian_refresh_interval = 100;
p.hessian_min_eig = 1;
p.hessian_safety = 1e-10;

p.record_interval = 0.5;
p.max_time = 50;
end
