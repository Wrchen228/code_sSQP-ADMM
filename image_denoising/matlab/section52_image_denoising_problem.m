function problem = section52_image_denoising_problem(caseName, opts)
%SECTION52_IMAGE_DENOISING_PROBLEM Build the paper Section 5.2 test problem.
%
% The original model in the paper is:
%   min_x,y  sum_i q(|(∇x)_i|) + lambdaY * ||y||_1
%   s.t.     x + B y = c
%
% This helper returns the constrained model together with an optional
% augmented objective for monitoring ADMM progress.
%
% The returned struct contains the clean image, synthetic observation,
% texton kernel, and function handles for objective and gradients.
if nargin < 1 || isempty(caseName)
    caseName = 'bird';
end
if nargin < 2
    opts = struct();
end

cfg = local_case_config(caseName, opts);

xTrueRaw = load_paper_image(cfg.cleanPath, [], false);
if ~isempty(cfg.cleanCropRect)
    xTrueRaw = local_crop_rect(xTrueRaw, cfg.cleanCropRect);
end
if any(size(xTrueRaw) ~= cfg.imageSize)
    xTrue = imresize(xTrueRaw, cfg.imageSize, 'bicubic');
else
    xTrue = xTrueRaw;
end
xTrue = min(max(xTrue, 0), 1);
if isfield(cfg, 'kernelArray') && ~isempty(cfg.kernelArray)
    kernel = cfg.kernelArray;
else
    kernel = load_section52_texton(cfg.kernelPath, cfg.kernelSize, struct( ...
        'normalize', cfg.kernelNormalize, ...
        'cropRect', cfg.kernelCropRect, ...
        'cropContent', cfg.kernelCropContent, ...
        'preserveAspect', cfg.kernelPreserveAspect));
end
kernel = cfg.kernelGain * kernel;
if isfield(cfg, 'kernelMeanZero') && cfg.kernelMeanZero
    kernel = kernel - mean(kernel(:));
end
if cfg.usePaperObservation
    c = load_paper_image(cfg.observationPath, cfg.imageSize, true);
    yTrue = [];
else
    yTrue = section52_random_coefficients(size(xTrue), cfg);
    noise = apply_circular_convolution(yTrue, kernel);
    if isfield(cfg, 'targetNoisyPSNR') && ~isempty(cfg.targetNoisyPSNR) && isfinite(cfg.targetNoisyPSNR)
        noise = local_rescale_noise_to_psnr(xTrue, noise, cfg.targetNoisyPSNR);
        yTrue = local_backproject_scaled_noise(noise, kernel, yTrue);
    end
    c = xTrue + noise;
end

problem = struct();
problem.name = cfg.name;
problem.caseName = cfg.name;
problem.imageSize = size(xTrue);
problem.kernelSize = size(kernel);
problem.kernelNormalize = cfg.kernelNormalize;
problem.kernelGain = cfg.kernelGain;
problem.priorType = cfg.priorType;
problem.priorDelta = cfg.priorDelta;
problem.lambdaY = cfg.lambdaY;
problem.rho = cfg.rho;
problem.noiseScale = cfg.noiseScale;
problem.seed = cfg.seed;
problem.xTrue = xTrue;
problem.yTrue = yTrue;
problem.cleanImage = xTrue;
problem.noisyImage = c;
problem.noisyImageClipped = min(max(c, 0), 1);
problem.noiseImage = problem.noisyImage - problem.cleanImage;
problem.kernel = kernel;
problem.numComponents = numel(xTrue);
problem.numPixels = numel(xTrue);
problem.numSamples = numel(xTrue);
kernelF = fft2(kernel, size(xTrue, 1), size(xTrue, 2));
problem.operatorNormB2 = max(abs(kernelF(:)).^2);
% For q(t)=log(1+t^2/delta^2), |q''(t)| is bounded by 2/delta^2.
% Since the periodic forward-difference operator satisfies ||D||^2 <= 8,
% the gradient Lipschitz bound of sum_i q(||(Dx)_i||) is 16/delta^2.
problem.L = max(16 / max(cfg.priorDelta^2, eps), 1);

problem.B = @(z) apply_circular_convolution(z, kernel);
problem.Bt = @(z) apply_circular_adjoint(z, kernel);
problem.residual = @(x, y) x + problem.B(y) - c;
problem.initial = local_consistent_initialization(problem, c);

problem.f_value = @(x) section52_regularizer_value(x, cfg.priorType, cfg.priorDelta);
problem.g_value = @(y) cfg.lambdaY * sum(abs(y(:)));
problem.objective = @(x, y) problem.f_value(x) + problem.g_value(y);
problem.augmentedObjective = @(x, y, lambda, beta) section52_augmented_objective(problem, x, y, lambda, beta);

problem.f_grad = @(x, batchMask) prior_gradient(x, cfg.priorType, cfg.priorDelta, batchMask);
problem.full_grad = @(x) prior_gradient(x, cfg.priorType, cfg.priorDelta, []);
problem.batch_grad = @(x, batchMask) prior_gradient(x, cfg.priorType, cfg.priorDelta, batchMask);
problem.component_field = @(x) section52_component_field(x, cfg.priorType, cfg.priorDelta);
problem.field_to_grad = @(field) -divergence_periodic(field(:,:,1), field(:,:,2));
problem.grad_x = @(x, y, batchMask) problem.f_grad(x, batchMask) + cfg.rho * problem.residual(x, y);
problem.grad_y = @(x, y) cfg.lambdaY * sign(y) + cfg.rho * problem.Bt(problem.residual(x, y));

problem.sampleMask = @(batchSize) sample_batch_mask(size(xTrue), batchSize);
problem.fullMask = true(size(xTrue));
problem.metadata = struct();
problem.metadata.paper = '(2024-MC) Stochastic alternating structure-adapted proximal gradient descent method with variance reduction for nonconvex nonsmooth optimization';
problem.metadata.section = '5.2';
problem.metadata.description = 'Stationary-noise image denoising with logarithmic nonconvex gradient prior and linear convolution constraint.';
problem.metadata.noiseDistribution = cfg.noiseDistribution;
problem.metadata.usePaperObservation = cfg.usePaperObservation;
problem.metadata.saturationFraction = mean(c(:) < 0 | c(:) > 1);
problem.metadata.minObservation = min(c(:));
problem.metadata.maxObservation = max(c(:));
if isfield(cfg, 'targetNoisyPSNR')
    problem.metadata.targetNoisyPSNR = cfg.targetNoisyPSNR;
else
    problem.metadata.targetNoisyPSNR = [];
end

end

function cfg = local_case_config(caseName, opts)
caseName = lower(caseName);
rootDir = fileparts(fileparts(mfilename('fullpath')));
rawDir = fullfile(rootDir, 'data', 'raw');
paperDir = fullfile(rootDir, 'data', 'paper_pdf');

cfg = struct();
cfg.name = caseName;
cfg.normalizeKernel = 'fro';
cfg.kernelNormalize = 'fro';
cfg.kernelGain = 1;
cfg.priorType = 'logquad';
cfg.priorDelta = 0.05;
cfg.lambdaY = 1e-2;
cfg.rho = 1;
cfg.seed = 0;
cfg.usePaperObservation = false;
cfg.noiseDistribution = 'laplace';
cfg.noiseMean = 0;

switch caseName
    case 'bird'
        cfg.cleanPath = fullfile(rawDir, 'bird.png');
        cfg.imageSize = [128, 128];
        cfg.cleanCropRect = [52, 8, 128, 128];
        cfg.kernelPath = fullfile(paperDir, 'img-003.png');
        cfg.kernelCropRect = [1, 1, 340, 110];
        cfg.kernelCropContent = false;
        cfg.kernelPreserveAspect = true;
        cfg.observationPath = fullfile(paperDir, 'img-005.png');
        cfg.kernelSize = [9, 9];
        cfg.kernelArray = local_elliptic_gaussian([9, 9], 2.1, 1.0);
        cfg.kernelGain = 1;
        cfg.kernelMeanZero = true;
        cfg.lambdaY = 0.1;
        cfg.noiseScale = 0.2;
        cfg.priorDelta = 0.0200;
        cfg.targetNoisyPSNR = 24.0;
    case 'clock'
        cfg.cleanPath = fullfile(rawDir, 'clock_128x128.png');
        cfg.imageSize = [128, 128];
        cfg.cleanCropRect = [];
        cfg.kernelPath = fullfile(paperDir, 'img-003.png');
        cfg.kernelCropRect = [1, 1, 340, 110];
        cfg.kernelCropContent = false;
        cfg.kernelPreserveAspect = true;
        cfg.observationPath = '';
        cfg.kernelSize = [9, 9];
        cfg.kernelArray = local_elliptic_gaussian([9, 9], 2.1, 1.0);
        cfg.kernelGain = 1;
        cfg.kernelMeanZero = true;
        cfg.lambdaY = 12;
        cfg.noiseScale = 0.2;
        cfg.priorDelta = 0.05;
        cfg.targetNoisyPSNR = 20.0;
    case 'boat'
        cfg.cleanPath = fullfile(rawDir, 'boat.png');
        cfg.imageSize = [256, 256];
        cfg.cleanCropRect = [];
        cfg.kernelPath = fullfile(paperDir, 'img-009.png');
        cfg.kernelCropRect = [1, 1, 150, 120];
        cfg.kernelCropContent = false;
        cfg.kernelPreserveAspect = true;
        cfg.observationPath = fullfile(paperDir, 'img-011.png');
        cfg.kernelSize = [41, 41];
        cfg.kernelArray = local_oriented_gaussian([41, 41], 9.0, 2.2, -35);
        cfg.kernelGain = 1;
        cfg.kernelMeanZero = true;
        cfg.lambdaY = 0.05;
        cfg.noiseScale = 0.6;
        cfg.targetNoisyPSNR = 21.8;
    case 'peppers'
        cfg.cleanPath = fullfile(rawDir, 'peppers.png');
        cfg.imageSize = [512, 512];
        cfg.cleanCropRect = [];
        cfg.kernelPath = fullfile(paperDir, 'img-006.png');
        cfg.kernelCropRect = [1, 1, 180, 240];
        cfg.kernelCropContent = false;
        cfg.kernelPreserveAspect = true;
        cfg.observationPath = fullfile(paperDir, 'img-008.png');
        cfg.kernelSize = [41, 31];
        cfg.kernelArray = local_oriented_gaussian([41, 31], 10.0, 2.5, 60);
        cfg.kernelGain = 1;
        cfg.kernelMeanZero = true;
        cfg.lambdaY = 0.05;
        cfg.noiseScale = 0.4;
        cfg.targetNoisyPSNR = 20.0;
    case 'male'
        % USC-SIPI 5.3.01 Male: native 1024-by-1024 8-bit grayscale image.
        % Keep the large-image noise model consistent with peppers so that
        % the added experiment isolates the effect of image scale.
        cfg.cleanPath = fullfile(rawDir, 'Male.tiff');
        cfg.imageSize = [1024, 1024];
        cfg.cleanCropRect = [];
        cfg.kernelPath = fullfile(paperDir, 'img-006.png');
        cfg.kernelCropRect = [1, 1, 180, 240];
        cfg.kernelCropContent = false;
        cfg.kernelPreserveAspect = true;
        cfg.observationPath = '';
        cfg.kernelSize = [41, 31];
        cfg.kernelArray = local_oriented_gaussian([41, 31], 10.0, 2.5, 60);
        cfg.kernelGain = 1;
        cfg.kernelMeanZero = true;
        cfg.lambdaY = 0.05;
        cfg.noiseScale = 0.4;
        cfg.priorDelta = 0.2;
        cfg.targetNoisyPSNR = 20.0;
    otherwise
        error('Unknown Section 5.2 case: %s', caseName);
end

fields = fieldnames(opts);
for i = 1:numel(fields)
    cfg.(fields{i}) = opts.(fields{i});
end

if ~isfield(cfg, 'kernelNormalize') || isempty(cfg.kernelNormalize)
    cfg.kernelNormalize = 'fro';
end
if ~isfield(cfg, 'cleanCropRect')
    cfg.cleanCropRect = [];
end
if ~isfield(cfg, 'kernelCropRect')
    cfg.kernelCropRect = [];
end
if ~isfield(cfg, 'kernelCropContent')
    cfg.kernelCropContent = true;
end
if ~isfield(cfg, 'kernelPreserveAspect')
    cfg.kernelPreserveAspect = false;
end
if ~isfield(cfg, 'kernelArray')
    cfg.kernelArray = [];
end
if ~isfield(cfg, 'kernelGain') || isempty(cfg.kernelGain)
    cfg.kernelGain = 1;
end
if ~isfield(cfg, 'kernelMeanZero') || isempty(cfg.kernelMeanZero)
    cfg.kernelMeanZero = false;
end
if ~isfield(cfg, 'priorType') || isempty(cfg.priorType)
    cfg.priorType = 'logquad';
end
if ~isfield(cfg, 'priorDelta') || isempty(cfg.priorDelta)
    cfg.priorDelta = 0.05;
end
if ~isfield(cfg, 'lambdaY') || isempty(cfg.lambdaY)
    cfg.lambdaY = 1e-2;
end
if ~isfield(cfg, 'rho') || isempty(cfg.rho)
    cfg.rho = 1;
end
if ~isfield(cfg, 'seed') || isempty(cfg.seed)
    cfg.seed = 0;
end
if ~isfield(cfg, 'usePaperObservation') || isempty(cfg.usePaperObservation)
    cfg.usePaperObservation = false;
end
if ~isfield(cfg, 'noiseDistribution') || isempty(cfg.noiseDistribution)
    cfg.noiseDistribution = 'laplace';
end
if ~isfield(cfg, 'noiseMean') || isempty(cfg.noiseMean)
    cfg.noiseMean = 0;
end
if ~isfield(cfg, 'targetNoisyPSNR')
    cfg.targetNoisyPSNR = [];
end
end

function val = section52_regularizer_value(x, priorType, delta)
[gx, gy] = gradient_periodic(x);
mag = sqrt(gx.^2 + gy.^2);
[q, ~] = image_prior(mag, priorType, delta);
val = sum(q(:));
end

function field = section52_component_field(x, priorType, delta)
[dx, dy] = gradient_periodic(x);
mag = sqrt(dx.^2 + dy.^2);
[~, coeff] = image_prior(mag, priorType, delta);
field = zeros([size(x), 2]);
field(:,:,1) = coeff .* dx;
field(:,:,2) = coeff .* dy;
end

function val = section52_augmented_objective(problem, x, y, lambda, beta)
r = problem.residual(x, y);
val = problem.objective(x, y) - sum(lambda(:) .* r(:)) + 0.5 * beta * sum(r(:).^2);
end

function y = section52_random_coefficients(sz, cfg)
seed = cfg.seed + 1;
switch lower(cfg.noiseDistribution)
    case 'laplace'
        y = laplace_rand(sz, cfg.noiseMean, cfg.noiseScale, seed);
    case 'gaussian'
        rng(seed, 'twister');
        y = cfg.noiseMean + cfg.noiseScale * randn(sz);
    case 'uniform'
        rng(seed, 'twister');
        halfWidth = sqrt(3) * cfg.noiseScale;
        y = cfg.noiseMean + halfWidth * (2 * rand(sz) - 1);
    otherwise
        error('Unknown noiseDistribution: %s', cfg.noiseDistribution);
end
end

function noiseScaled = local_rescale_noise_to_psnr(xTrue, noise, targetPsnr)
err = noise(:);
baseNorm = norm(err);
if baseNorm <= eps
    noiseScaled = noise;
    return;
end
mseTarget = 10^(-targetPsnr / 10);
targetNorm = sqrt(numel(xTrue) * mseTarget);
scale = targetNorm / baseNorm;
noiseScaled = scale * noise;
end

function yScaled = local_backproject_scaled_noise(noise, kernel, yRef)
kernelF = fft2(kernel, size(noise, 1), size(noise, 2));
noiseF = fft2(noise);
den = abs(kernelF).^2;
mask = den > 1e-10;
yF = zeros(size(noiseF));
yF(mask) = noiseF(mask) ./ kernelF(mask);
yScaled = real(ifft2(yF));
if nargin >= 3 && ~isempty(yRef)
    yScaled = yScaled + 0 * yRef;
end
end

function out = local_crop_rect(img, rect)
rect = round(rect);
x1 = max(1, rect(1));
y1 = max(1, rect(2));
x2 = min(size(img, 2), x1 + rect(3) - 1);
y2 = min(size(img, 1), y1 + rect(4) - 1);
out = img(y1:y2, x1:x2);
end

function K = local_elliptic_gaussian(sz, sigmaX, sigmaY)
nx = sz(2);
ny = sz(1);
[X, Y] = meshgrid(1:nx, 1:ny);
cx = (nx + 1) / 2;
cy = (ny + 1) / 2;
X = X - cx;
Y = Y - cy;
K = exp(-0.5 * ((X / sigmaX).^2 + (Y / sigmaY).^2));
K = K / max(norm(K, 'fro'), eps);
end

function K = local_oriented_gaussian(sz, sigmaMajor, sigmaMinor, angleDeg)
nx = sz(2);
ny = sz(1);
[X, Y] = meshgrid(1:nx, 1:ny);
cx = (nx + 1) / 2;
cy = (ny + 1) / 2;
X = X - cx;
Y = Y - cy;
theta = angleDeg * pi / 180;
U = cos(theta) * X + sin(theta) * Y;
V = -sin(theta) * X + cos(theta) * Y;
K = exp(-0.5 * ((U / sigmaMajor).^2 + (V / sigmaMinor).^2));
K = K / max(norm(K, 'fro'), eps);
end

function init = local_consistent_initialization(problem, c)
% Build a gentle warm start:
%   - keep x0 close to the observed image so its PSNR is not destroyed
%   - initialize y0 with a small nonzero estimate from the clipping residual
% This avoids both the unrealistic y0 = 0 choice and the overly aggressive
% fully-feasible start that can move x0 far away from the image manifold.
x0 = problem.noisyImageClipped;
seedResidual = c - x0;
L0 = max(problem.operatorNormB2, eps);
initShrinkFactor = 0.2;
y0 = local_soft_threshold(problem.Bt(seedResidual) / L0, initShrinkFactor * problem.lambdaY / L0);

init = struct();
init.x0 = x0;
init.y0 = y0;
init.description = 'Warm start: x0 = clipped noisy image, y0 = lightly soft-thresholded estimate from clipping residual.';
end

function z = local_soft_threshold(x, t)
z = sign(x) .* max(abs(x) - t, 0);
end
