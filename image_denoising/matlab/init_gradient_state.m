function state = init_gradient_state(method, x0, problem, opts)
%INIT_GRADIENT_STATE Initialize variance-reduced estimator state.
state.method = lower(method);
state.step = 0;
state.n = problem.numSamples;
state.batchSize = opts.batchSize;
state.snapshotPeriod = opts.snapshotPeriod;
switch state.method
    case 'full'
        state.fullGrad = problem.full_grad(x0);
    case 'sgd'
        state.fullGrad = [];
    case 'svrg'
        state.snapshot = x0;
        state.snapshotGrad = problem.full_grad(x0);
    case 'saga'
        state.pStore = problem.component_field(x0);
    case 'sarah'
        state.prevX = x0;
        state.prevField = problem.component_field(x0);
        state.vPrev = problem.full_grad(x0);
    otherwise
        error('Unknown method: %s', method);
end
end
