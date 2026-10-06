function [x, y, history] = section52_ssqp_admm_solver(problem, opts)
%SECTION52_SSQP_ADMM_SOLVER Section 5.2 solver following the paper code flow.
%
% The Alg.1 branches (SARAH, SVRG and SAGA) follow the supplied LASSO
% package in Desktop/实验1: one mini-batch per iteration, a shared
% adaptive SQP-ADMM step, and estimator-specific gradient updates.
%
% The current Section 5.2 model is:
%   min_{x,y} f(x) + lambdaY * ||y||_1
%   s.t.      x + B y = c
%
% For this problem, A = I and phi(x,y) = x + B y - c.
if nargin < 2
    opts = struct();
end
if ~isfield(opts, 'method'), opts.method = 'sarah'; end
if ~isfield(opts, 'mode'), opts.mode = 'constrained'; end
if ~isfield(opts, 'maxIter'), opts.maxIter = 10000; end
if ~isfield(opts, 'maxTime'), opts.maxTime = inf; end
if ~isfield(opts, 'batchSize'), opts.batchSize = max(256, round(0.02 * problem.numSamples)); end
if ~isfield(opts, 'numInnerIters')
    if isfield(opts, 'snapshotPeriod')
        opts.numInnerIters = opts.snapshotPeriod;
    else
        opts.numInnerIters = 10;
    end
end
if ~isfield(opts, 'beta0'), opts.beta0 = 0.02; end
if ~isfield(opts, 's'), opts.s = 0.95; end
if ~isfield(opts, 'delta'), opts.delta = 0.5; end
if ~isfield(opts, 'varrho'), opts.varrho = 0.8; end
if ~isfield(opts, 'epsXi'), opts.epsXi = 0.8; end
if ~isfield(opts, 'zeta1'), opts.zeta1 = 0.75; end
if ~isfield(opts, 'zeta2'), opts.zeta2 = 0.1; end
if ~isfield(opts, 'tau'), opts.tau = 1e-3; end
if ~isfield(opts, 'Py'), opts.Py = 0.05; end
if ~isfield(opts, 'xi0'), opts.xi0 = 1.0; end
if ~isfield(opts, 'psnrTarget'), opts.psnrTarget = inf; end
if ~isfield(opts, 'x0'), opts.x0 = problem.initial.x0; end
if ~isfield(opts, 'y0'), opts.y0 = problem.initial.y0; end
if ~isfield(opts, 'lambda0'), opts.lambda0 = zeros(size(problem.cleanImage)); end
if ~isfield(opts, 'verbose'), opts.verbose = true; end
% The image prior is written as a pixel sum, whereas the reference LASSO
% implementation uses an average finite-sum loss.  Solve the equivalent
% normalized primal problem internally so beta, xi and alpha have the same
% scale as in Algorithm 1; report the original objective in history.
if ~isfield(opts, 'objectiveScale'), opts.objectiveScale = 1 / problem.numSamples; end
if ~isscalar(opts.objectiveScale) || opts.objectiveScale <= 0 || ~isfinite(opts.objectiveScale)
    error('objectiveScale must be a finite positive scalar.');
end
if isfield(opts, 'beta0Original')
    if ~isscalar(opts.beta0Original) || opts.beta0Original <= 0 || ~isfinite(opts.beta0Original)
        error('beta0Original must be a finite positive scalar.');
    end
    % Convert a paper-scale penalty to the internally normalized objective.
    opts.beta0 = opts.beta0Original * opts.objectiveScale;
end

if any(strcmpi(opts.method, {'svrgadmm', 'svrg_admm'}))
    [x, y, history] = local_svrgadmm_solver(problem, opts);
    return;
end

if any(strcmpi(opts.method, {'bsarahsadmm', 'bsarah'}))
    [x, y, history] = local_bsarahsadmm_solver(problem, opts);
    return;
end

if any(strcmpi(opts.method, {'sadmm_timevarying', 'sadmm-timevarying', 'sadmm_tv'}))
    [x, y, history] = local_timevarying_sadmm_solver(problem, opts);
    return;
end

if any(strcmpi(opts.method, {'sarah', 'svrg', 'saga', 'full'}))
    % Keep Alg.1's image implementation synchronized with the validated
    % LASSO package in Desktop/实验1: one mini-batch per iteration and a
    % shared adaptive SQP-ADMM step for all variance-reduction variants.
    [x, y, history] = local_alg1_image_solver(problem, opts);
    return;
end

x = opts.x0;
y = opts.y0;
lambda = opts.lambda0;
beta = opts.beta0;
tildeXi = opts.xi0;


capacity = max(200, opts.maxIter);
history = local_alloc_history(capacity);
history = local_record_initial_history(problem, history, x, y, lambda, beta, opts.objectiveScale);

normA2 = 1; % A = I for Section 5.2.
n = problem.numSamples;
k = 2;
tic;

while k <= opts.maxIter
    if toc >= opts.maxTime
        break;
    end

    iterNo = k - 1;
    gammaK = 1 / (iterNo + 1)^2;
    betaOuter = beta;
    w = x;
    xSnapshot = x;
    fullGradSnapshot = problem.full_grad(xSnapshot);
    hOuter = local_sampled_hessian_diag(problem, xSnapshot, opts);
    hTilde = hOuter + betaOuter * normA2;

    permIdx = randperm(n);
    vPrev = fullGradSnapshot;
    wPrev = w;
    sagaState = [];
    if strcmpi(opts.method, 'saga')
        sagaState = local_saga_init(problem, xSnapshot);
    end

    lastAlpha = NaN;
    lastAlphaBar = NaN;
    lastAlphaTilde = NaN;
    lastXiBar = NaN;
    lastTau = NaN;
    lastYBefore = local_y_augmented_value(problem, w, y, lambda, betaOuter);
    lastYAfter = lastYBefore;
    lastQBefore = NaN;
    lastQAfter = NaN;
    lastDeltaQ = NaN;

    for j = 1:opts.numInnerIters
        if toc >= opts.maxTime
            break;
        end

        yPrev = y;
        lastYBefore = local_y_augmented_value(problem, w, yPrev, lambda, betaOuter);
        y = local_update_y_inner(problem, w, yPrev, lambda, betaOuter, opts);
        lastYAfter = local_y_augmented_value(problem, w, y, lambda, betaOuter);

        idxBatch = local_batch_indices(permIdx, j, opts.batchSize, n);
        switch lower(opts.method)
            case 'sarah'
                mask = false(problem.imageSize);
                mask(idxBatch) = true;
                nb = numel(idxBatch);
                gradW = (n / nb) * problem.batch_grad(w, mask);
                gradWPrev = (n / nb) * problem.batch_grad(wPrev, mask);
                v = gradW - gradWPrev + vPrev;
                vPrev = v;
                wPrev = w;
            case 'svrg'
                mask = false(problem.imageSize);
                mask(idxBatch) = true;
                nb = numel(idxBatch);
                gradW = (n / nb) * problem.batch_grad(w, mask);
                gradSnap = (n / nb) * problem.batch_grad(xSnapshot, mask);
                v = gradW - gradSnap + fullGradSnapshot;
            case 'full'
                v = fullGradSnapshot;
            case 'saga'
                nb = numel(idxBatch);
                [v, sagaState] = local_saga_gradient(problem, w, sagaState, idxBatch, n, nb);
            otherwise
                error('Unsupported method in rewritten flow: %s', opts.method);
        end

        linearTerm = v - lambda + betaOuter * problem.residual(w, y);
        dbarX = -linearTerm ./ max(hTilde, eps);

        res1 = problem.residual(w, y);
        epsilon = sum(res1(:).^2) - sum((res1(:) + dbarX(:)).^2);
        aVal = epsilon - opts.zeta2 * sum(dbarX(:).^2);
        aDag = 0;
        if aVal > 0
            aDag = 1 / aVal;
        end

        dHd = sum(hOuter(:) .* (dbarX(:).^2));
        tildeBeta = 2 * aDag * (sum((v(:) - lambda(:)) .* dbarX(:)) + opts.zeta1 * dHd);

        term = sum((v(:) - lambda(:)) .* dbarX(:)) + dHd;
        if term <= 0
            beta = max(min((1 + gammaK) * betaOuter, tildeBeta), betaOuter);
        else
            beta = max(betaOuter / (1 + gammaK), tildeBeta);
        end

        deltaQ = -sum((v(:) - lambda(:)) .* dbarX(:)) + 0.5 * beta * epsilon;
        xiBar = deltaQ / max(sum(dbarX(:).^2), 1e-12);
        if tildeXi <= xiBar
            tildeXiNext = tildeXi;
        else
            tildeXiNext = min((1 - opts.epsXi) * tildeXi, xiBar);
        end
        tildeXi = tildeXiNext;

        alphaBar = min(2 * (1 - opts.delta) * xiBar / (problem.L + beta * normA2), 1);
        alphaTilde = min(2 * (1 - opts.delta) * tildeXi / (problem.L + beta * normA2), 1);
        alpha = max(alphaTilde, min(alphaTilde + opts.varrho, alphaBar));

        lastQBefore = local_x_q_value(problem, w, y, lambda, beta, zeros(size(dbarX)), v);
        wNext = w + alpha * dbarX;
        lastQAfter = local_x_q_value(problem, w, y, lambda, beta, alpha * dbarX, v);
        lastDeltaQ = lastQBefore - lastQAfter;

        w = wNext;
        if strcmpi(opts.method, 'svrg')
                currentDualStep = opts.s / (1 + 0.01 * iterNo);
        else
            currentDualStep = opts.s;
        end
        lambda = lambda - currentDualStep * beta * problem.residual(w, y);

        lastAlpha = alpha;
        lastAlphaBar = alphaBar;
        lastAlphaTilde = alphaTilde;
        lastXiBar = xiBar;
        lastTau = min(hOuter(:));
    end

    x = w;

    if k > numel(history.obj)
        history = local_grow_history(history, numel(history.obj));
    end

    history.fval(k) = problem.f_value(x);
    history.gval(k) = problem.g_value(y);
    history.obj(k) = history.fval(k) + history.gval(k);
    history.yALFBefore(k) = lastYBefore;
    history.yALFAfter(k) = lastYAfter;
    history.xqBefore(k) = lastQBefore;
    history.xqAfter(k) = lastQAfter;
    history.xDeltaQ(k) = lastDeltaQ;
    history.residual(k) = norm(problem.residual(x, y), 'fro') / max(1, norm(problem.noisyImage, 'fro'));
    history.alpha(k) = lastAlpha;
    history.alphaBar(k) = lastAlphaBar;
    history.alphaTilde(k) = lastAlphaTilde;
    history.beta(k) = beta;
    history.gradNorm(k) = norm(vPrev, 'fro');
    history.tau(k) = lastTau;
    history.xiBar(k) = lastXiBar;
    history.xiTilde(k) = tildeXi;
    history.time(k) = toc;
    history.psnr(k) = psnr_gray(x, problem.xTrue);


    if opts.verbose && (iterNo == 1 || mod(iterNo, 10) == 0)
        fprintf('outer %d: obj=%.6e res=%.3e alpha=%.3f beta=%.3f psnr=%.3f\n', ...
            iterNo, history.obj(k), history.residual(k), history.alpha(k), history.beta(k), history.psnr(k));
    end

    k = k + 1;
end

lastIdx = find(history.time > 0, 1, 'last');
if isempty(lastIdx)
    lastIdx = 1;
end
history = local_trim_history(history, lastIdx);
history.finalTime = history.time(end);
if toc >= opts.maxTime
    history.stopReason = 'maxTime';
end
% PSNR is an offline diagnostic only.  Never use the clean image to select
% an iterate or to control the optimization trajectory.
[history.bestPSNR, history.bestIter] = max(history.psnr);
history.finalPSNR = history.psnr(end);
end

function [x, y, history] = local_alg1_image_solver(problem, opts)
%LOCAL_ALG1_IMAGE_SOLVER Image counterpart of SQPADMM_{sarah,svrg,saga}.
opts = local_alg1_image_options(problem, opts);
x = opts.x0;
y = opts.y0;
lambda = opts.lambda0;
beta = opts.beta0;
xi = opts.xi0;
n = problem.numSamples;
k = 0;
previousX = x;
previousGradient = opts.objectiveScale * problem.full_grad(x);
snapshot = x;
fullSnapshotGradient = previousGradient;
sagaState = [];
if strcmpi(opts.method, 'saga')
    sagaState = local_saga_init(problem, x, opts.objectiveScale);
end
hessianState = struct();
% Start with a bounded history and grow it only if needed.  This lets
% maxIter be safely larger than the wall-clock budget without a large
% up-front memory allocation.
history = local_alloc_history(max(200, min(opts.maxIter, 10000)));
history = local_record_initial_history(problem, history, x, y, lambda, beta);
tic;

while k < opts.maxIter && toc < opts.maxTime
    idxBatch = randperm(n, opts.batchSize);
    batchMask = false(problem.imageSize);
    batchMask(idxBatch) = true;
    nb = numel(idxBatch);

    % Use the same deterministic refresh schedule as SVRG-/SARAH-ADMM.
    refreshNow = (k == 0) || mod(k, opts.alg1RefreshPeriod) == 0;

    % These three estimators use exactly the same logic as the LASSO code.
    switch lower(opts.method)
        case 'sarah'
            if refreshNow
                gEst = opts.objectiveScale * problem.full_grad(x);
            else
                gEst = opts.objectiveScale * (n / nb) * (problem.batch_grad(x, batchMask) ...
                    - problem.batch_grad(previousX, batchMask)) + previousGradient;
            end
        case 'svrg'
            if refreshNow && k > 0
                snapshot = x;
                fullSnapshotGradient = opts.objectiveScale * problem.full_grad(snapshot);
            end
            gEst = opts.objectiveScale * (n / nb) * (problem.batch_grad(x, batchMask) ...
                - problem.batch_grad(snapshot, batchMask)) + fullSnapshotGradient;
        case 'saga'
            [gEst, sagaState] = local_saga_gradient(problem, x, sagaState, idxBatch, n, nb, opts.objectiveScale);
        case 'full'
            gEst = opts.objectiveScale * problem.full_grad(x);
    end

    [xNext, yNext, lambdaNext, betaNext, xiNext, alpha, tauK, hessianState, stepInfo] = ...
        local_alg1_image_step(problem, x, y, lambda, gEst, beta, xi, k, batchMask, opts, hessianState);
    if toc >= opts.maxTime
        break;
    end

    previousX = x;
    previousGradient = gEst;
    x = xNext;
    y = yNext;
    lambda = lambdaNext;
    beta = betaNext;
    xi = xiNext;
    k = k + 1;
    hIdx = k + 1;
    if hIdx > numel(history.obj), history = local_grow_history(history, numel(history.obj)); end

    history.fval(hIdx) = problem.f_value(x);
    history.gval(hIdx) = problem.g_value(y);
    history.obj(hIdx) = history.fval(hIdx) + history.gval(hIdx);
    history.yALFBefore(hIdx) = stepInfo.yBefore;
    history.yALFAfter(hIdx) = stepInfo.yAfter;
    history.xqBefore(hIdx) = stepInfo.xBefore;
    history.xqAfter(hIdx) = stepInfo.xAfter;
    history.xDeltaQ(hIdx) = stepInfo.xBefore - stepInfo.xAfter;
    history.residual(hIdx) = norm(problem.residual(x, y), 'fro') / max(1, norm(problem.noisyImage, 'fro'));
    history.alpha(hIdx) = alpha;
    history.alphaBar(hIdx) = stepInfo.alphaBar;
    history.alphaTilde(hIdx) = stepInfo.alphaTilde;
    history.beta(hIdx) = beta;
    history.gradNorm(hIdx) = norm(gEst, 'fro');
    history.tau(hIdx) = tauK;
    history.xiBar(hIdx) = stepInfo.xiBar;
    history.xiTilde(hIdx) = xi;
    history.time(hIdx) = toc;
    history.psnr(hIdx) = psnr_gray(x, problem.xTrue);
    if opts.verbose && (k == 1 || mod(k, 10) == 0)
        fprintf('Alg.1-%s %d: obj=%.6e res=%.3e alpha=%.3e beta=%.3e\n', ...
            lower(opts.method), k, history.obj(hIdx), history.residual(hIdx), alpha, beta);
    end
end

lastIdx = k + 1;
history = local_trim_history(history, lastIdx);
history.finalTime = history.time(end);
if toc >= opts.maxTime
    history.stopReason = 'maxTime';
end
[history.bestPSNR, history.bestIter] = max(history.psnr);
history.finalPSNR = history.psnr(end);
end

function opts = local_alg1_image_options(problem, opts)
if ~isfield(opts, 'refreshProb'), opts.refreshProb = 1 / 50; end
if ~isfield(opts, 'alg1RefreshPeriod')
    % Preserve compatibility with old probability-based configurations.
    opts.alg1RefreshPeriod = max(1, round(1 / max(opts.refreshProb, eps)));
end
if ~isfield(opts, 'K0'), opts.K0 = 500; end
if ~isfield(opts, 'gammaMax'), opts.gammaMax = 0.5; end
if ~isfield(opts, 'gammaScale'), opts.gammaScale = 1.5; end
if ~isfield(opts, 'gammaPower'), opts.gammaPower = 2; end
% Match the supplied Alg.1 implementation: reuse the curvature model for a
% short period to avoid recomputing the expensive sampled Hessian every step.
if ~isfield(opts, 'hessianRefreshInterval'), opts.hessianRefreshInterval = 5; end
% The image code solves the objective scaled by objectiveScale=1/n, while
% the Algorithm 1 implementation specifies the spectral lower bound in the
% original objective scale.  Scale that lower bound as well; otherwise an
% H_k >= I request would make the normalized image step n times too small.
if isfield(opts, 'hessianMinEigOriginal')
    opts.hessianMinEig = max(0, opts.hessianMinEigOriginal) * opts.objectiveScale;
elseif ~isfield(opts, 'hessianMinEig')
    opts.hessianMinEig = opts.objectiveScale;
end
if ~isfield(opts, 'hessianSafety'), opts.hessianSafety = 1e-10; end
if ~isfield(opts, 'alg1PcgTol'), opts.alg1PcgTol = 1e-4; end
if ~isfield(opts, 'alg1PcgMaxIter'), opts.alg1PcgMaxIter = 50; end
if ~isfield(opts, 'alg1UseSpectralShift'), opts.alg1UseSpectralShift = false; end
if ~isfield(opts, 'alg1HessianEigTol'), opts.alg1HessianEigTol = 1e-2; end
if ~isfield(opts, 'alg1HessianEigMaxIter'), opts.alg1HessianEigMaxIter = 20; end
if ~isfield(opts, 'alg1YMaxIter'), opts.alg1YMaxIter = 100; end
if ~isfield(opts, 'alg1YTol'), opts.alg1YTol = 1e-4; end
% The beta-branch threshold uses hat_vartheta_k=(1-zeta1)/(k+1), which
% requires zeta1 in [0,1] to satisfy hat_vartheta_k in [0,1-zeta1].
if ~isscalar(opts.zeta1) || ~isfinite(opts.zeta1) ...
        || opts.zeta1 < 0 || opts.zeta1 > 1
    error('opts.zeta1 must lie in [0,1].');
end
opts.batchSize = min(problem.numSamples, max(1, round(opts.batchSize)));
opts.refreshProb = min(1, max(0, opts.refreshProb));
opts.alg1RefreshPeriod = max(1, round(opts.alg1RefreshPeriod));
opts.hessianRefreshInterval = max(1, round(opts.hessianRefreshInterval));
end

function [xNext, yNext, lambdaNext, betaNext, xiNext, alpha, tauK, state, info] = ...
    local_alg1_image_step(problem, x, y, lambda, gEst, beta, xi, k, batchMask, opts, state)
% One-to-one image adaptation of adaptive_sqp_admm_step.m.
yBefore = local_y_augmented_value(problem, x, y, lambda, beta, opts.objectiveScale);
yNext = local_alg1_y_update(problem, x, y, lambda, beta, opts);
yAfter = local_y_augmented_value(problem, x, yNext, lambda, beta, opts.objectiveScale);

refresh = ~isfield(state, 'hessian') || mod(k, opts.hessianRefreshInterval) == 0;
if refresh
    state.hessian = local_image_batch_hessian_state(problem, x, batchMask, opts);
end
tauK = state.hessian.tauK;
residual = problem.residual(x, yNext);
linearTerm = gEst - lambda + beta * residual;
% The image prior has a non-diagonal and possibly indefinite Hessian.  Solve
% the shifted SQP system matrix-free rather than treating its gradient flux
% coefficient as a diagonal Hessian entry.
imageSize = size(x);
systemAction = @(v) local_image_system_vector(v, imageSize, state.hessian, beta);
[dVector, pcgFlag] = pcg(systemAction, -linearTerm(:), opts.alg1PcgTol, opts.alg1PcgMaxIter);
dX = reshape(dVector, imageSize);
if pcgFlag ~= 0 || any(~isfinite(dX(:)))
    % The spectral shift makes this fallback positive and keeps the iteration
    % well defined if PCG reaches its iteration limit.
    dX = -linearTerm / max(beta + tauK, eps);
end
A_d = dX; % A=I for x+B*y-c=0.
epsilonK = sum(residual(:).^2) - sum((residual(:) + A_d(:)).^2);
varthetaK = local_alg1_vartheta(k, opts);
hD = local_image_hessian_action(dX, state.hessian);
dHd = sum(dX(:) .* hD(:));
if norm(residual, 'fro') > varthetaK * min(1, norm(A_d, 'fro'))
    denominator = epsilonK - opts.zeta2 * sum(A_d(:).^2);
    if denominator ~= 0
        betaTrial = 2 * ((sum((gEst(:) - lambda(:)) .* dX(:)) + opts.zeta1 * dHd) / denominator);
    else
        betaTrial = 0;
    end
else
    betaTrial = 0;
end

gammaK = opts.gammaMax / (1 + k / opts.gammaScale)^opts.gammaPower;
% R_k and ||d_x^k||_{H_k}^2 in the stated adaptive penalty rule.
Rk = sum((gEst(:) - lambda(:)) .* dX(:)) + dHd;
hatVarthetaK = (1 - opts.zeta1) / (k + 1);
betaBranchThreshold = hatVarthetaK * min(1, dHd);
if Rk <= betaBranchThreshold
    betaNext = max(min((1 + gammaK) * beta, betaTrial), beta);
elseif k < opts.K0
    betaNext = max(beta / (1 + gammaK), betaTrial);
else
    betaNext = beta;
end
deltaQ = -sum((gEst(:) - lambda(:)) .* dX(:)) + 0.5 * betaNext * epsilonK;
dNormSq = sum(dX(:).^2);
if dNormSq == 0
    xiBar = inf;
else
    xiBar = deltaQ / dNormSq;
end
if xi <= xiBar
    xiNext = xi;
else
    xiNext = min((1 - opts.epsXi) * xi, xiBar);
end
scaledL = opts.objectiveScale * problem.L;
alphaBar = min(2 * (1 - opts.delta) * xiBar / (scaledL + betaNext), 1);
alphaTilde = min(2 * (1 - opts.delta) * xiNext / (scaledL + betaNext), 1);
alpha = max(alphaTilde, min(alphaTilde + opts.varrho, alphaBar));
xNext = x + alpha * dX;
xBefore = local_x_q_value(problem, x, yNext, lambda, beta, zeros(size(dX)), gEst, opts.objectiveScale);
xAfter = local_x_q_value(problem, x, yNext, lambda, beta, alpha * dX, gEst, opts.objectiveScale);
% As in the LASSO code, use beta_k rather than beta_{k+1} in the dual step.
lambdaNext = lambda - opts.s * beta * problem.residual(xNext, yNext);
if betaNext <= 0 || ~isfinite(betaNext) || ~isfinite(alpha)
    error('Alg.1 image update produced an invalid beta or alpha.');
end
info = struct('yBefore', yBefore, 'yAfter', yAfter, 'xBefore', xBefore, ...
    'xAfter', xAfter, 'xiBar', xiBar, 'alphaBar', alphaBar, 'alphaTilde', alphaTilde);
end

function varthetaK = local_alg1_vartheta(k, opts)
% Evaluate the threshold sequence used in the tilde-beta safeguard.
if ~isfield(opts, 'varthetaMode') || isempty(opts.varthetaMode)
    mode = 'inverse_sqrt';
else
    mode = lower(char(opts.varthetaMode));
end
switch mode
    case {'zero', '0'}
        varthetaK = 0;
    case {'inverse', '1/(k+1)'}
        varthetaK = 1 / (k + 1);
    case {'inverse_sqrt', '1/sqrt(k+1)'}
        varthetaK = 1 / sqrt(k + 1);
    case {'constant', 'fixed'}
        if ~isfield(opts, 'varthetaConstant') || isempty(opts.varthetaConstant)
            varthetaK = 0.5;
        elseif ~isscalar(opts.varthetaConstant) || ~isfinite(opts.varthetaConstant) ...
                || opts.varthetaConstant < 0
            error('opts.varthetaConstant must be a finite nonnegative scalar.');
        else
            varthetaK = opts.varthetaConstant;
        end
    otherwise
        error('Unknown opts.varthetaMode: %s', mode);
end
end

function y = local_alg1_y_update(problem, x, yPrev, lambda, beta, opts)
% P_k=beta*Py*I, matching adaptive_sqp_admm_step.m in the LASSO package.
% Use monotone proximal-gradient iterations here.  An unprotected FISTA
% extrapolation can increase the y-subproblem ALF and create residual spikes.
Lips = beta * (problem.operatorNormB2 + opts.Py);
step = 1 / max(Lips, eps);
y = yPrev;
for it = 1:opts.alg1YMaxIter
    residual = problem.residual(x, y);
    grad = -problem.Bt(lambda) + beta * problem.Bt(residual) + beta * opts.Py * (y - yPrev);
    yNext = soft_threshold(y - step * grad, step * opts.objectiveScale * problem.lambdaY);

    % Test the proximal-gradient mapping, not merely the step length.  This
    % gives an approximate minimizer of the same y subproblem that is solved
    % in closed form in the LASSO implementation.
    residualNext = problem.residual(x, yNext);
    gradNext = -problem.Bt(lambda) + beta * problem.Bt(residualNext) + ...
        beta * opts.Py * (yNext - yPrev);
    yPG = soft_threshold(yNext - step * gradNext, step * opts.objectiveScale * problem.lambdaY);
    pgResidual = norm((yNext(:) - yPG(:)) / step) / sqrt(numel(yNext));
    if pgResidual <= opts.alg1YTol
        y = yNext;
        return;
    end
    y = yNext;
end
end

function state = local_image_batch_hessian_state(problem, x, batchMask, opts)
% Exact Hessian action of the sampled log-quadratic image prior.  At one
% pixel, its 2-by-2 curvature is a*I-b*[dx;dy]*[dx,dy], so a scalar diagonal
% surrogate would lose the nonconvex radial-curvature term.
[dx, dy] = gradient_periodic(x);
nb = nnz(batchMask);
scale = problem.numSamples / max(nb, 1);
delta2 = max(problem.priorDelta, eps)^2;
mag2 = dx.^2 + dy.^2;

switch lower(problem.priorType)
    case 'logquad'
        scale = opts.objectiveScale * scale;
        state.a = scale * batchMask .* (2 ./ (delta2 + mag2));
        state.b = scale * batchMask .* (4 ./ (delta2 + mag2).^2);
        % The two local eigenvalues of M are a and a-b*||grad x||^2.
        % This local bound is retained as a guaranteed fallback if the
        % matrix-free Lanczos estimate below cannot be obtained.
        localMinEigenvalue = min(state.a(:) - state.b(:) .* mag2(:));
        fallbackNegativeCurvatureBound = max(0, -8 * localMinEigenvalue);
    otherwise
        error('Alg.1 image Hessian is implemented for priorType=''logquad'' only.');
end

state.dx = dx;
state.dy = dy;
negativeCurvatureBound = fallbackNegativeCurvatureBound;
if opts.alg1UseSpectralShift
    eigOpts = struct('tol', opts.alg1HessianEigTol, ...
        'maxit', opts.alg1HessianEigMaxIter, 'issym', true, 'isreal', true);
    try
        imageSize = size(x);
        hessianVector = @(v) local_image_unshifted_hessian_vector(v, imageSize, state);
        [~, spectralValue, eigFlag] = eigs(hessianVector, problem.numSamples, 1, 'sa', eigOpts);
        lambdaMin = spectralValue(1, 1);
        if eigFlag == 0 && isfinite(lambdaMin)
            negativeCurvatureBound = max(0, -real(lambdaMin));
        end
    catch
        % Keep the certified local-curvature bound when ARPACK does not
        % converge within the requested lightweight iteration budget.
    end
end
state.tauK = opts.hessianMinEig + negativeCurvatureBound + opts.hessianSafety;
end

function Hv = local_image_hessian_action(v, state)
% Applies H_k = nabla^2 f_B(x^k) + tau_k I without forming an n-by-n matrix.
Hv = local_image_unshifted_hessian_action(v, state) + state.tauK * v;
end

function Hv = local_image_unshifted_hessian_action(v, state)
% Applies only the sampled Hessian, before the positive spectral shift.
[vx, vy] = gradient_periodic(v);
directionalGradient = state.dx .* vx + state.dy .* vy;
hx = state.a .* vx - state.b .* state.dx .* directionalGradient;
hy = state.a .* vy - state.b .* state.dy .* directionalGradient;
Hv = -divergence_periodic(hx, hy);
end

function Hv = local_image_unshifted_hessian_vector(v, imageSize, state)
imageVector = reshape(v, imageSize);
Hv = local_image_unshifted_hessian_action(imageVector, state);
Hv = Hv(:);
end

function Hv = local_image_system_vector(v, imageSize, state, beta)
% PCG accepts vectors, whereas the image Hessian is naturally evaluated on
% two-dimensional arrays.
imageVector = reshape(v, imageSize);
Hv = local_image_hessian_action(imageVector, state) + beta * imageVector;
Hv = Hv(:);
end

function history = local_alloc_history(capacity)
fields = {'obj','fval','gval','yALFBefore','yALFAfter','xqBefore','xqAfter', ...
    'xDeltaQ','residual','alpha','alphaBar','alphaTilde','beta','psnr', ...
    'gradNorm','tau','xiBar','xiTilde','time'};
for i = 1:numel(fields)
    history.(fields{i}) = zeros(capacity, 1);
end
history.stopReason = 'maxIter';
history.finalTime = 0;
history.bestPSNR = NaN;
history.bestIter = NaN;
history.finalPSNR = NaN;
end

function history = local_record_initial_history(problem, history, x, y, lambda, beta, objectiveScale)
if nargin < 7, objectiveScale = 1; end
history.fval(1) = problem.f_value(x);
history.gval(1) = problem.g_value(y);
history.obj(1) = history.fval(1) + history.gval(1);
r = problem.residual(x, y);
history.yALFBefore(1) = objectiveScale * problem.g_value(y) ...
    - sum(lambda(:) .* r(:)) + 0.5 * beta * sum(r(:).^2);
history.yALFAfter(1) = history.yALFBefore(1);
history.xqBefore(1) = NaN;
history.xqAfter(1) = NaN;
history.xDeltaQ(1) = NaN;
history.residual(1) = norm(problem.residual(x, y), 'fro') / max(1, norm(problem.noisyImage, 'fro'));
history.alpha(1) = NaN;
history.alphaBar(1) = NaN;
history.alphaTilde(1) = NaN;
history.beta(1) = beta;
history.psnr(1) = psnr_gray(x, problem.xTrue);
history.gradNorm(1) = NaN;
history.tau(1) = NaN;
history.xiBar(1) = NaN;
history.xiTilde(1) = NaN;
history.time(1) = 0;
end

function history = local_grow_history(history, oldCap)
newCap = 2 * oldCap;
fields = fieldnames(history);
for i = 1:numel(fields)
    val = history.(fields{i});
    if isnumeric(val) && isvector(val) && numel(val) == oldCap
        history.(fields{i})(newCap, 1) = 0;
    end
end
end

function history = local_trim_history(history, lastIdx)
fields = fieldnames(history);
for i = 1:numel(fields)
    val = history.(fields{i});
    if isnumeric(val) && isvector(val) && numel(val) >= lastIdx
        history.(fields{i}) = val(1:lastIdx);
    end
end
end

function idxBatch = local_batch_indices(permIdx, j, batchSize, n)
sIdx = mod((j - 1) * batchSize, n) + 1;
eIdx = min(sIdx + batchSize - 1, n);
idxBatch = permIdx(sIdx:eIdx);
if numel(idxBatch) < batchSize
    idxBatch = [idxBatch, permIdx(1:(batchSize - numel(idxBatch)))];
end
end

function state = local_saga_init(problem, x, objectiveScale)
% Compact SAGA table for this image prior. Each finite-sum component is a
% local two-channel field value; the full x-gradient is obtained by applying
% the divergence once to the accumulated field.
if nargin < 3, objectiveScale = 1; end
state = struct();
state.field = problem.component_field(x);
state.gradSum = objectiveScale * problem.field_to_grad(state.field);
end

function [v, state] = local_saga_gradient(problem, x, state, idxBatch, n, nb, objectiveScale)
if nargin < 7, objectiveScale = 1; end
currentField = problem.component_field(x);
diffField = zeros(size(state.field));

oldPage = state.field(:,:,1);
newPage = currentField(:,:,1);
diffPage = zeros(problem.imageSize);
diffPage(idxBatch) = newPage(idxBatch) - oldPage(idxBatch);
diffField(:,:,1) = diffPage;
oldPage(idxBatch) = newPage(idxBatch);
state.field(:,:,1) = oldPage;

oldPage = state.field(:,:,2);
newPage = currentField(:,:,2);
diffPage = zeros(problem.imageSize);
diffPage(idxBatch) = newPage(idxBatch) - oldPage(idxBatch);
diffField(:,:,2) = diffPage;
oldPage(idxBatch) = newPage(idxBatch);
state.field(:,:,2) = oldPage;

diffGrad = problem.field_to_grad(diffField);
v = state.gradSum + objectiveScale * (n / nb) * diffGrad;
state.gradSum = state.gradSum + objectiveScale * diffGrad;
end

function hdiag = local_sampled_hessian_diag(problem, x, opts)
[dx, dy] = gradient_periodic(x);
mag = sqrt(dx.^2 + dy.^2);
[~, coeff] = image_prior(mag, problem.priorType, problem.priorDelta);
hdiag = 2 * coeff + circshift(coeff, 1, 1) + circshift(coeff, 1, 2);
hdiag = hdiag + opts.tau;
hdiag = max(hdiag, opts.tau);
end

function y = local_update_y_inner(problem, x, yPrev, lambda, betaK, opts)
Lips = betaK * problem.operatorNormB2 + opts.Py;
step = 1 / max(Lips, eps);
y = yPrev;
for it = 1:5
    r = problem.residual(x, y);
    grad = -problem.Bt(lambda) + betaK * problem.Bt(r) + opts.Py * (y - yPrev);
    yNext = soft_threshold(y - step * grad, step * problem.lambdaY);
    if norm(yNext(:) - y(:)) <= 1e-4 * max(1, norm(y(:)))
        y = yNext;
        break;
    end
    y = yNext;
end
end

function val = local_y_augmented_value(problem, x, y, lambda, betaK, objectiveScale)
if nargin < 6, objectiveScale = 1; end
r = problem.residual(x, y);
val = objectiveScale * problem.g_value(y) - sum(lambda(:) .* r(:)) + 0.5 * betaK * sum(r(:).^2);
end

function val = local_y_augmented_value_plus(problem, x, y, lambda, betaK, objectiveScale)
if nargin < 6, objectiveScale = 1; end
r = problem.residual(x, y);
val = objectiveScale * problem.g_value(y) + sum(lambda(:) .* r(:)) + 0.5 * betaK * sum(r(:).^2);
end

function val = local_x_q_value(problem, x, y, lambda, betaK, dx, gbar, objectiveScale)
if nargin < 9, objectiveScale = 1; end
r = problem.residual(x, y);
val = objectiveScale * problem.f_value(x) ...
    - sum(lambda(:) .* r(:)) ...
    + sum((gbar(:) - lambda(:)) .* dx(:)) ...
    + 0.5 * betaK * sum((r(:) + dx(:)).^2);
end

function z = soft_threshold(x, t)
z = sign(x) .* max(abs(x) - t, 0);
end

function y = local_update_y_inner_plus(problem, x, yPrev, lambda, betaK, opts)
Lips = betaK * problem.operatorNormB2 + opts.Py;
step = 1 / max(Lips, eps);
y = yPrev;
for it = 1:5
    r = problem.residual(x, y);
    grad = problem.Bt(lambda) + betaK * problem.Bt(r) + opts.Py * (y - yPrev);
    yNext = soft_threshold(y - step * grad, step * problem.lambdaY);
    if norm(yNext(:) - y(:)) <= 1e-4 * max(1, norm(y(:)))
        y = yNext;
        break;
    end
    y = yNext;
end
end

function y = local_update_y_inner_plus_plain(problem, x, yPrev, lambda, betaK)
Lips = betaK * problem.operatorNormB2;
step = 1 / max(Lips, eps);
y = yPrev;
for it = 1:10
    r = problem.residual(x, y);
    grad = problem.Bt(lambda) + betaK * problem.Bt(r);
    yNext = soft_threshold(y - step * grad, step * problem.lambdaY);
    if norm(yNext(:) - y(:)) <= 1e-5 * max(1, norm(y(:)))
        y = yNext;
        break;
    end
    y = yNext;
end
end

function [x, y, history] = local_timevarying_sadmm_solver(problem, opts)
%LOCAL_TIMEVARYING_SADMM_SOLVER Image adaptation of Experiment 1's
% Huang--Chen SADMM with eta_t=2*sqrt(t+1).  As in the source algorithm,
% use one uniformly sampled finite-sum component per iteration, a fixed
% penalty, and the effective decreasing x step 1/(eta_t+rho).
if ~isfield(opts, 'timeVaryingEtaScale'), opts.timeVaryingEtaScale = 2; end
if ~isfield(opts, 'timeVaryingEtaPower'), opts.timeVaryingEtaPower = 0.5; end
if ~isfield(opts, 'timeVaryingRho'), opts.timeVaryingRho = 0.1; end
if ~isfield(opts, 'timeVaryingYMaxIter'), opts.timeVaryingYMaxIter = 25; end
if ~isfield(opts, 'timeVaryingYTol'), opts.timeVaryingYTol = 1e-4; end
if ~isfield(opts, 'timeVaryingBatchFraction'), opts.timeVaryingBatchFraction = []; end
if ~isfield(opts, 'maxIter'), opts.maxIter = 100000; end
if ~isfield(opts, 'maxTime'), opts.maxTime = inf; end
if ~isfield(opts, 'x0'), opts.x0 = problem.initial.x0; end
if ~isfield(opts, 'y0'), opts.y0 = problem.initial.y0; end
if ~isfield(opts, 'lambda0'), opts.lambda0 = zeros(size(problem.cleanImage)); end
if ~isfield(opts, 'verbose'), opts.verbose = true; end
if ~isscalar(opts.timeVaryingEtaScale) || opts.timeVaryingEtaScale <= 0 || ...
        ~isfinite(opts.timeVaryingEtaScale)
    error('timeVaryingEtaScale must be a finite positive scalar.');
end
if ~isscalar(opts.timeVaryingEtaPower) || opts.timeVaryingEtaPower < 0 || ...
        opts.timeVaryingEtaPower > 1 || ~isfinite(opts.timeVaryingEtaPower)
    error('timeVaryingEtaPower must be a finite scalar in [0,1].');
end
if ~isscalar(opts.timeVaryingRho) || opts.timeVaryingRho <= 0 || ...
        ~isfinite(opts.timeVaryingRho)
    error('timeVaryingRho must be a finite positive scalar.');
end
if isempty(opts.timeVaryingBatchFraction)
    timeVaryingBatchSize = 1;
elseif ~isscalar(opts.timeVaryingBatchFraction) || ...
        opts.timeVaryingBatchFraction <= 0 || opts.timeVaryingBatchFraction > 1 || ...
        ~isfinite(opts.timeVaryingBatchFraction)
    error('timeVaryingBatchFraction must be empty or a finite scalar in (0,1].');
else
    timeVaryingBatchSize = max(1, round(opts.timeVaryingBatchFraction * problem.numSamples));
end

x = opts.x0;
y = opts.y0;
lambda = opts.lambda0;
rho = opts.timeVaryingRho;
n = problem.numSamples;
history = local_alloc_history(max(200, min(opts.maxIter, 10000)));
history = local_record_initial_history(problem, history, x, y, lambda, rho, opts.objectiveScale);
history.batchSize = timeVaryingBatchSize;
tic;
k = 1;
while k < opts.maxIter && toc < opts.maxTime
    t = k - 1;
    etaT = opts.timeVaryingEtaScale * (t + 1)^opts.timeVaryingEtaPower;

    % Experiment 1 uses one sample. For image finite sums an explicitly
    % configured spatial mini-batch avoids updating only one local pixel
    % neighborhood per expensive convolutional y step.
    sampleIds = randperm(n, timeVaryingBatchSize);
    sampleMask = false(problem.imageSize);
    sampleMask(sampleIds) = true;
    gSample = opts.objectiveScale * (n / timeVaryingBatchSize) * ...
        problem.batch_grad(x, sampleMask);

    yBefore = local_y_augmented_value(problem, x, y, lambda, rho, opts.objectiveScale);
    yNext = local_timevarying_y_step(problem, x, y, lambda, rho, opts);
    yAfter = local_y_augmented_value(problem, x, yNext, lambda, rho, opts.objectiveScale);

    % Exact Q=I surrogate minimizer for A=I and x+B*y-c=0.
    residual = problem.residual(x, yNext);
    xNext = x - (gSample - lambda + rho * residual) / (etaT + rho);
    lambdaNext = lambda - rho * problem.residual(xNext, yNext);
    if toc >= opts.maxTime
        break;
    end

    x = xNext;
    y = yNext;
    lambda = lambdaNext;
    k = k + 1;
    if k > numel(history.obj), history = local_grow_history(history, numel(history.obj)); end
    history.fval(k) = problem.f_value(x);
    history.gval(k) = problem.g_value(y);
    history.obj(k) = history.fval(k) + history.gval(k);
    history.yALFBefore(k) = yBefore;
    history.yALFAfter(k) = yAfter;
    history.xqBefore(k) = NaN; history.xqAfter(k) = NaN; history.xDeltaQ(k) = NaN;
    history.residual(k) = norm(problem.residual(x, y), 'fro') / ...
        max(1, norm(problem.noisyImage, 'fro'));
    history.alpha(k) = 1 / (etaT + rho);
    history.alphaBar(k) = history.alpha(k); history.alphaTilde(k) = history.alpha(k);
    history.beta(k) = rho;
    history.gradNorm(k) = norm(gSample, 'fro');
    history.tau(k) = etaT;
    history.xiBar(k) = NaN; history.xiTilde(k) = NaN;
    history.time(k) = toc;
    history.psnr(k) = psnr_gray(x, problem.xTrue);
    if opts.verbose && (k == 2 || mod(k - 1, 100) == 0)
        fprintf('SADMM-time-varying %d: obj=%.6e res=%.3e eta=%.3e\n', ...
            k - 1, history.obj(k), history.residual(k), etaT);
    end
end

history = local_trim_history(history, k);
history.finalTime = history.time(end);
if toc >= opts.maxTime, history.stopReason = 'maxTime'; end
[history.bestPSNR, history.bestIter] = max(history.psnr);
history.finalPSNR = history.psnr(end);
end

function y = local_timevarying_y_step(problem, x, y, lambda, rho, opts)
% Convolutional counterpart of Experiment 1's exact l1 proximal y step.
step = 1 / max(rho * problem.operatorNormB2, eps);
z = y;
momentum = 1;
for it = 1:opts.timeVaryingYMaxIter
    grad = -problem.Bt(lambda) + rho * problem.Bt(problem.residual(x, z));
    yNext = soft_threshold(z - step * grad, ...
        step * opts.objectiveScale * problem.lambdaY);
    gradNext = -problem.Bt(lambda) + rho * problem.Bt(problem.residual(x, yNext));
    yPG = soft_threshold(yNext - step * gradNext, ...
        step * opts.objectiveScale * problem.lambdaY);
    pgResidual = norm((yNext(:) - yPG(:)) / step) / sqrt(numel(yNext));
    if pgResidual <= opts.timeVaryingYTol
        y = yNext;
        return;
    end
    momentumNext = (1 + sqrt(1 + 4 * momentum^2)) / 2;
    z = yNext + ((momentum - 1) / momentumNext) * (yNext - y);
    y = yNext;
    momentum = momentumNext;
end
end

function [x, y, history] = local_svrgadmm_solver(problem, opts)
%LOCAL_SVRGADMM_SOLVER NC2021 Algorithm 1 with the SVRG estimator.
%
% Let z=c-B*y.  The image model is then min H(x)+F(z) subject to x-z=0,
% with F(z)=min_y {lambda*||y||_1 : z=c-B*y}.  Thus A=I in Algorithm 1
% of Bian et al. (Inverse Problems, 2021); its exact z-step is computed
% below through the equivalent convolutional l1 y-subproblem.
opts = local_nc2021_options(problem, opts);
x = opts.x0;
y = opts.y0;
u = opts.lambda0;
beta = opts.beta0;
    % tau is the denominator of the explicit x-step.  Keep SVRG and
    % SARAH independently tunable because their estimator variances differ.
    tau = opts.sadmmTauSVRG;
    sigma = opts.sadmmSigmaSVRG;
n = problem.numSamples;
snapshot = x;
fullSnapshot = opts.objectiveScale * problem.full_grad(snapshot);
bestX = x; bestY = y; bestPSNR = psnr_gray(x, problem.xTrue); bestIter = 1;
% Start with a bounded history and grow it only if needed.
history = local_alloc_history(max(200, min(opts.maxIter, 10000)));
history = local_record_initial_history(problem, history, x, y, u, beta, opts.objectiveScale);
tic;

for k = 2:opts.maxIter
    if toc >= opts.maxTime
        break;
    end
    iterNo = k - 1;
    % The initial snapshot is formed before the loop (iteration 0).
    % Thereafter all snapshot-based methods refresh at 10,20,30,...
    if mod(iterNo, opts.sadmmSnapshotPeriod) == 0
        snapshot = x;
        fullSnapshot = opts.objectiveScale * problem.full_grad(snapshot);
    end

    % Algorithm 1 (1.5a): z-step, evaluated through y with z=c-B*y.
    yBefore = local_y_augmented_value_plus(problem, x, y, u, beta, opts.objectiveScale);
    y = local_nc2021_y_step(problem, x, y, u, beta, opts);
    yAfter = local_y_augmented_value_plus(problem, x, y, u, beta, opts.objectiveScale);

    idxBatch = randperm(n, opts.batchSize);
    mask = false(problem.imageSize); mask(idxBatch) = true;
    nb = numel(idxBatch);
    v = opts.objectiveScale * (n / nb) * (problem.batch_grad(x, mask) - problem.batch_grad(snapshot, mask)) ...
        + fullSnapshot;

    % Algorithm 1 (1.5b)-(1.5c), with A=I and x-z=x+B*y-c.
    xNext = x - (v + u + beta * problem.residual(x, y)) / tau;
    u = u + sigma * beta * problem.residual(xNext, y);
    x = xNext;

    history = local_nc2021_record(problem, history, k, x, y, beta, tau, v, yBefore, yAfter, toc);
    [bestX, bestY, bestPSNR, bestIter] = local_update_best(x, y, history.psnr(k), ...
        bestX, bestY, bestPSNR, bestIter, k);
    if opts.verbose && (iterNo == 1 || mod(iterNo, 20) == 0)
        fprintf('SVRG-ADMM %d: obj=%.6e res=%.3e\n', iterNo, history.obj(k), history.residual(k));
    end
    if isfinite(opts.psnrTarget) && history.psnr(k) >= opts.psnrTarget
        history.stopReason = 'psnrTarget'; break;
    end
end
[x, y, history] = local_nc2021_finish(history, x, y, bestX, bestY, bestPSNR, bestIter, opts, toc);
end

function [x, y, history] = local_bsarahsadmm_solver(problem, opts)
%LOCAL_BSAR AHSADMM_SOLVER NC2021 Algorithm 1 with the SARAH estimator.
opts = local_nc2021_options(problem, opts);
x = opts.x0;
y = opts.y0;
u = opts.lambda0;
beta = opts.beta0;
    % SARAH accumulates recursive estimator error, so it normally needs a
    % more conservative explicit x-step than the SVRG baseline.
    tau = opts.sadmmTauSARAH;
    sigma = opts.sadmmSigmaSARAH;
n = problem.numSamples;
v = opts.objectiveScale * problem.full_grad(x); % SARAH: v_0 = full gradient.
xPrevious = x;
bestX = x; bestY = y; bestPSNR = psnr_gray(x, problem.xTrue); bestIter = 1;
% Start with a bounded history and grow it only if needed.
history = local_alloc_history(max(200, min(opts.maxIter, 10000)));
history = local_record_initial_history(problem, history, x, y, u, beta, opts.objectiveScale);
tic;

for k = 2:opts.maxIter
    if toc >= opts.maxTime, break; end
    iterNo = k - 1;
    % For a controlled comparison, use the same deterministic period as
    % SVRG and Alg.1 rather than a Bernoulli refresh event.
    if mod(iterNo, opts.sadmmSarahRefreshPeriod) == 0
        v = opts.objectiveScale * problem.full_grad(x);
    else
        idxBatch = randperm(n, opts.batchSize);
        mask = false(problem.imageSize); mask(idxBatch) = true;
        nb = numel(idxBatch);
        v = v + opts.objectiveScale * (n / nb) * ...
            (problem.batch_grad(x, mask) - problem.batch_grad(xPrevious, mask));
    end

    yBefore = local_y_augmented_value_plus(problem, x, y, u, beta, opts.objectiveScale);
    y = local_nc2021_y_step(problem, x, y, u, beta, opts);
    yAfter = local_y_augmented_value_plus(problem, x, y, u, beta, opts.objectiveScale);
    xPrevious = x;
    xNext = x - (v + u + beta * problem.residual(x, y)) / tau;
    u = u + sigma * beta * problem.residual(xNext, y);
    x = xNext;

    history = local_nc2021_record(problem, history, k, x, y, beta, tau, v, yBefore, yAfter, toc);
    [bestX, bestY, bestPSNR, bestIter] = local_update_best(x, y, history.psnr(k), ...
        bestX, bestY, bestPSNR, bestIter, k);
    if opts.verbose && (iterNo == 1 || mod(iterNo, 20) == 0)
        fprintf('SARAH-ADMM %d: obj=%.6e res=%.3e\n', iterNo, history.obj(k), history.residual(k));
    end
    if isfinite(opts.psnrTarget) && history.psnr(k) >= opts.psnrTarget
        history.stopReason = 'psnrTarget'; break;
    end
end
[x, y, history] = local_nc2021_finish(history, x, y, bestX, bestY, bestPSNR, bestIter, opts, toc);
end

function opts = local_nc2021_options(problem, opts)
% Fixed parameters required by Algorithm 1: 2*tau >= beta*||A||^2, A=I.
if ~isfield(opts, 'beta0'), opts.beta0 = 1; end
if ~isfield(opts, 'sadmmSigma')
    if isfield(opts, 'sigma'), opts.sadmmSigma = opts.sigma; else, opts.sadmmSigma = 0.95; end
end
if ~isfield(opts, 'sadmmTau')
    if isfield(opts, 'tauBSARAH'), opts.sadmmTau = opts.tauBSARAH; else, opts.sadmmTau = max(1, opts.beta0 / 2); end
end
if ~isfield(opts, 'sadmmTauSVRG'), opts.sadmmTauSVRG = opts.sadmmTau; end
if ~isfield(opts, 'sadmmTauSARAH'), opts.sadmmTauSARAH = opts.sadmmTau; end
if ~isfield(opts, 'sadmmSigmaSVRG'), opts.sadmmSigmaSVRG = opts.sadmmSigma; end
if ~isfield(opts, 'sadmmSigmaSARAH'), opts.sadmmSigmaSARAH = opts.sadmmSigma; end

% The paper's algebraic lower bound alone does not account for the
% normalized curvature of H. Warn about a practically unstable x step.
normalizedL = problem.L / problem.numSamples;
tauNames = {'sadmmTauSVRG', 'sadmmTauSARAH'};
sigmaNames = {'sadmmSigmaSVRG', 'sadmmSigmaSARAH'};
for i = 1:numel(tauNames)
    tauValue = opts.(tauNames{i});
    sigmaValue = opts.(sigmaNames{i});
    if ~isscalar(tauValue) || ~isfinite(tauValue) || tauValue < opts.beta0 / 2
        error('NC2021 SADMM requires %s >= beta0/2 because A=I.', tauNames{i});
    end
    if tauValue < normalizedL + opts.beta0
        warning('section52:SADMMTauTooSmall', ...
            ['%s=%g is below normalizedL+beta=%g. ', ...
             'The explicit SADMM x-step may become unstable.'], ...
            tauNames{i}, tauValue, normalizedL + opts.beta0);
    end
    if ~isscalar(sigmaValue) || sigmaValue <= 0 || sigmaValue > 1
        error('NC2021 SADMM requires %s in (0,1].', sigmaNames{i});
    end
end
if ~isfield(opts, 'sadmmSnapshotPeriod'), opts.sadmmSnapshotPeriod = 10; end
if ~isfield(opts, 'sadmmSarahRefreshPeriod'), opts.sadmmSarahRefreshPeriod = 10; end
if ~isfield(opts, 'sadmmYMaxIter'), opts.sadmmYMaxIter = 500; end
if ~isfield(opts, 'sadmmYTol'), opts.sadmmYTol = 1e-6; end
if ~isfield(opts, 'x0'), opts.x0 = problem.initial.x0; end
if ~isfield(opts, 'y0'), opts.y0 = problem.initial.y0; end
if ~isfield(opts, 'lambda0'), opts.lambda0 = zeros(size(problem.cleanImage)); end
if ~isfield(opts, 'maxIter'), opts.maxIter = 5000; end
if ~isfield(opts, 'maxTime'), opts.maxTime = inf; end
if ~isfield(opts, 'batchSize'), opts.batchSize = max(256, round(0.02 * problem.numSamples)); end
opts.batchSize = min(problem.numSamples, max(1, round(opts.batchSize)));
opts.sadmmSnapshotPeriod = max(1, round(opts.sadmmSnapshotPeriod));
opts.sadmmSarahRefreshPeriod = max(1, round(opts.sadmmSarahRefreshPeriod));
if ~isfield(opts, 'verbose'), opts.verbose = true; end
if ~isfield(opts, 'psnrTarget'), opts.psnrTarget = inf; end
end

function y = local_nc2021_y_step(problem, x, y, u, beta, opts)
% z-step in y variables: min lambda||y||_1+beta/2||B*y-d||^2.
d = problem.noisyImage - x - u / beta;
Lips = beta * max(problem.operatorNormB2, eps);
step = 1 / Lips;
z = y;
t = 1;
for it = 1:opts.sadmmYMaxIter
    grad = beta * problem.Bt(problem.B(z) - d);
    yNext = soft_threshold(z - step * grad, step * opts.objectiveScale * problem.lambdaY);
    % The proximal-gradient mapping is zero precisely at a z-subproblem
    % minimizer.  It is a meaningful KKT certificate even with FISTA.
    gradNext = beta * problem.Bt(problem.B(yNext) - d);
    yPG = soft_threshold(yNext - step * gradNext, step * opts.objectiveScale * problem.lambdaY);
    pgResidual = norm((yNext(:) - yPG(:)) / step) / sqrt(numel(yNext));
    if pgResidual <= opts.sadmmYTol
        y = yNext;
        return;
    end
    tNext = (1 + sqrt(1 + 4 * t^2)) / 2;
    z = yNext + ((t - 1) / tNext) * (yNext - y);
    y = yNext;
    t = tNext;
end
end

function history = local_nc2021_record(problem, history, k, x, y, beta, tau, v, yBefore, yAfter, elapsed)
if k > numel(history.obj), history = local_grow_history(history, numel(history.obj)); end
history.fval(k) = problem.f_value(x);
history.gval(k) = problem.g_value(y);
history.obj(k) = history.fval(k) + history.gval(k);
history.yALFBefore(k) = yBefore;
history.yALFAfter(k) = yAfter;
history.xqBefore(k) = NaN; history.xqAfter(k) = NaN; history.xDeltaQ(k) = NaN;
history.residual(k) = norm(problem.residual(x, y), 'fro') / max(1, norm(problem.noisyImage, 'fro'));
history.alpha(k) = 1 / tau; history.alphaBar(k) = 1 / tau; history.alphaTilde(k) = 1 / tau;
history.beta(k) = beta; history.gradNorm(k) = norm(v, 'fro'); history.tau(k) = tau;
history.xiBar(k) = NaN; history.xiTilde(k) = NaN;
history.time(k) = elapsed; history.psnr(k) = psnr_gray(x, problem.xTrue);
end

function [bestX, bestY, bestPSNR, bestIter] = local_update_best(x, y, currentPSNR, bestX, bestY, bestPSNR, bestIter, k)
if currentPSNR > bestPSNR
    bestX = x; bestY = y; bestPSNR = currentPSNR; bestIter = k;
end
end

function [x, y, history] = local_nc2021_finish(history, ~, ~, bestX, bestY, bestPSNR, bestIter, opts, elapsed)
lastIdx = find(history.time > 0, 1, 'last');
if isempty(lastIdx), lastIdx = 1; end
history = local_trim_history(history, lastIdx);
history.finalTime = history.time(end);
if elapsed >= opts.maxTime && ~strcmp(history.stopReason, 'psnrTarget')
    history.stopReason = 'maxTime';
end
history.bestPSNR = bestPSNR; history.bestIter = bestIter; history.finalPSNR = history.psnr(end);
% Curves use the final iterates; the return value remains the best-PSNR
% iterate so the recovery panel follows the experiment's visual protocol.
x = bestX; y = bestY;
end
