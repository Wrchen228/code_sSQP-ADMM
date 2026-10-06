function p = Set_ijcnn_Params(p)
%SET_IJCNN_PARAMS Initial parameters tailored to the normalized IJCNN data.
if nargin < 1
    p = struct();
end

% SSQP-ADMM parameters
p.K0 = 3000;
p.s = 0.8;
p.delta = 0.8;
p.varrho = 0.60;
p.epsilon_xi = 0.1;
p.zeta1 = 0.9;
p.zeta2 = 0.4;
p.beta = 0.10;
p.xi0 = 0.10;
p.Py = 0.05;
p.refresh_prob = 1 / 100;
p.batch_size = 64;
p.num_inner_iters = round(1 / p.refresh_prob);
p.gamma_max = 0.12;
p.gamma_warmup = 0;
p.gamma_scale = 6;
p.gamma_power = 2;
p.parameter_trace_max_iters = 5000;
p.parameter_trace_change_tol = 1e-5;
p.parameter_trace_tail = 5;
p.hessian_refresh_interval = 5;
p.hessian_min_eig = 1;
p.hessian_safety = 1e-10;
p.record_interval = 0.5;
p.max_time = 50;
end
