function [v, state, info] = estimate_gradient(x, problem, state)
%ESTIMATE_GRADIENT Build the stochastic gradient estimator v_k.
state.step = state.step + 1;
info = struct('mask', [], 'scale', 1);
switch state.method
    case 'full'
        v = problem.full_grad(x);
        info.mask = problem.fullMask;
    case 'sgd'
        mask = sample_batch_mask(size(x), state.batchSize);
        v = (state.n / state.batchSize) * problem.batch_grad(x, mask);
        info.mask = mask;
        info.scale = state.n / state.batchSize;
    case 'svrg'
        if mod(state.step - 1, state.snapshotPeriod) == 0
            state.snapshot = x;
            state.snapshotGrad = problem.full_grad(x);
        end
        mask = sample_batch_mask(size(x), state.batchSize);
        v = (state.n / state.batchSize) * (problem.batch_grad(x, mask) - problem.batch_grad(state.snapshot, mask)) + state.snapshotGrad;
        info.mask = mask;
        info.scale = state.n / state.batchSize;
    case 'saga'
        mask = sample_batch_mask(size(x), state.batchSize);
        currentField = problem.component_field(x);
        mask3 = repmat(mask, [1, 1, size(currentField, 3)]);
        vField = state.pStore + (state.n / state.batchSize) * (currentField - state.pStore) .* mask3;
        v = problem.field_to_grad(vField);
        state.pStore(mask3) = currentField(mask3);
        info.mask = mask;
        info.scale = state.n / state.batchSize;
    case 'sarah'
        if mod(state.step - 1, state.snapshotPeriod) == 0
            state.prevX = x;
            state.prevField = problem.component_field(x);
            state.vPrev = problem.full_grad(x);
        end
        mask = sample_batch_mask(size(x), state.batchSize);
        currentField = problem.component_field(x);
        prevField = problem.component_field(state.prevX);
        mask3 = repmat(mask, [1, 1, size(currentField, 3)]);
        deltaField = (currentField - prevField) .* mask3;
        v = state.vPrev + (state.n / state.batchSize) * problem.field_to_grad(deltaField);
        state.prevX = x;
        state.prevField = currentField;
        state.vPrev = v;
        info.mask = mask;
        info.scale = state.n / state.batchSize;
    otherwise
        error('Unknown method: %s', state.method);
end
end
