function results = run_section52_image_denoising(opts)
%RUN_SECTION52_IMAGE_DENOISING Prepare or solve the paper Section 5.2 cases.
%
% If opts.solverFcn is provided, it is called as:
%   [xHat, yHat, history] = opts.solverFcn(problem, opts.solverOpts)
%
% Otherwise the function only builds the test problems and saves previews.
if nargin < 1
    opts = struct();
end
if ~isfield(opts, 'cases')
    opts.cases = {'bird', 'boat', 'peppers'};
end
if ~isfield(opts, 'solverFcn')
    opts.solverFcn = @section52_ssqp_admm_solver;
end
if ~isfield(opts, 'solverOpts')
    opts.solverOpts = struct();
end
if ~isfield(opts, 'methods')
    opts.methods = {'sadmm_timevarying', 'svrgadmm', 'bsarah', 'svrg', 'saga', 'sarah'};
end
if ~isfield(opts, 'batchFractions')
    opts.batchFractions = [];
end
if ~isfield(opts, 'plot')
    opts.plot = true;
end
if ~isfield(opts, 'saveOutputs')
    opts.saveOutputs = true;
end
if ~isfield(opts, 'plotCurves')
    opts.plotCurves = true;
end
if ~isfield(opts, 'plotAdaptiveParameters')
    opts.plotAdaptiveParameters = true;
end
if ~isfield(opts, 'plotDecomposition')
    opts.plotDecomposition = false;
end
if ~isfield(opts, 'plotObjectiveBreakdown')
    opts.plotObjectiveBreakdown = false;
end
if ~isfield(opts, 'plotNoiseSummary')
    opts.plotNoiseSummary = true;
end
if ~isfield(opts, 'noiseSummaryName')
    opts.noiseSummaryName = 'section52_noise_summary';
end
if ~isfield(opts, 'randomSeed')
    opts.randomSeed = [];
end

rootDir = fileparts(fileparts(mfilename('fullpath')));
outDir = fullfile(rootDir, 'results', 'section52');
if opts.saveOutputs && ~exist(outDir, 'dir')
    mkdir(outDir);
end

results = struct();
for k = 1:numel(opts.cases)
    caseName = lower(opts.cases{k});
    problem = section52_image_denoising_problem(caseName, opts);

    caseResult = struct();
    caseResult.problem = problem;
    caseResult.methods = opts.methods;
    caseResult.batchFractions = opts.batchFractions;
    caseResult.runs = struct();
    caseResult.histories = struct();
    caseResult.legendLabels = struct();

    if ~isempty(opts.solverFcn)
        runKeys = {};
        if isempty(opts.batchFractions)
            batchFractions = NaN;
        else
            batchFractions = opts.batchFractions(:).';
        end
        for b = 1:numel(batchFractions)
            frac = batchFractions(b);
            for m = 1:numel(opts.methods)
                methodName = lower(opts.methods{m});
                solverOpts = opts.solverOpts;
                solverOpts.method = methodName;
                if ~isempty(opts.randomSeed)
                    % Give every method the same reproducible batch stream at
                    % the start of a case.  Their subsequent draws may differ
                    % by estimator, but rerunning the experiment is stable.
                    rng(opts.randomSeed + 1000 * (k - 1), 'twister');
                end
                if ~isnan(frac)
                    solverOpts.batchSize = max(1, round(frac * problem.numSamples));
                end
                runKey = section52_run_key(methodName, frac);
                [xHat, yHat, history] = opts.solverFcn(problem, solverOpts);
                ByHat = problem.B(yHat);
                residualImage = problem.noisyImage - xHat - ByHat;
                if ~isfield(history, 'batchSize') || isempty(history.batchSize)
                    if isfield(solverOpts, 'batchSize') && ~isempty(solverOpts.batchSize)
                        history.batchSize = solverOpts.batchSize;
                    else
                        history.batchSize = max(256, round(0.02 * problem.numSamples));
                    end
                end
                history.batchFraction = frac;
                caseResult.runs.(runKey).xHat = xHat;
                caseResult.runs.(runKey).yHat = yHat;
                caseResult.runs.(runKey).ByHat = ByHat;
                caseResult.runs.(runKey).residualImage = residualImage;
                caseResult.runs.(runKey).history = history;
                caseResult.histories.(runKey) = history;
                caseResult.legendLabels.(runKey) = section52_legend_label(methodName, frac, history.batchSize, problem.numSamples);
                runKeys{end + 1} = runKey; %#ok<AGROW>
            end
        end
        firstKey = runKeys{1};
        caseResult.xHat = caseResult.runs.(firstKey).xHat;
        caseResult.yHat = caseResult.runs.(firstKey).yHat;
        caseResult.history = caseResult.runs.(firstKey).history;
    end

    if opts.plot || opts.saveOutputs
        plot_section52_case(problem, caseResult, outDir, opts.plot, opts.saveOutputs);
        if opts.plotDecomposition
            plot_section52_decomposition(problem, caseResult, outDir, opts.plot, opts.saveOutputs);
        end
    end
    if opts.plotCurves && isfield(caseResult, 'history')
        plot_section52_combined_curves(problem, caseResult, outDir, ...
            opts.plotCurves, opts.saveOutputs, opts.solverOpts, opts.plotAdaptiveParameters);
        if opts.plotObjectiveBreakdown
            plot_section52_objective_breakdown(problem, caseResult, outDir, opts.plotCurves, opts.saveOutputs, opts.solverOpts);
        end
    end

    results.(caseName) = caseResult;
    if opts.saveOutputs
        save(fullfile(outDir, [caseName '.mat']), 'caseResult');
    end
end

if opts.plotNoiseSummary && (opts.plot || opts.saveOutputs)
    plot_section52_noise_summary(results, outDir, opts.plot, opts.saveOutputs, opts.noiseSummaryName);
end
end

function plot_section52_combined_curves(problem, caseResult, outDir, doPlot, doSave, solverOpts, showAdaptive)
% Five-panel horizontal summary with one shared legend.
if ~isfield(caseResult, 'histories') || isempty(fieldnames(caseResult.histories))
    return;
end
runKeys = section52_ordered_run_keys(caseResult, true);
algKeys = {};
for i = 1:numel(runKeys)
    nameParts = split(string(runKeys{i}), "_b");
    if any(strcmpi(char(nameParts(1)), {'sarah','svrg','saga'}))
        algKeys{end + 1} = runKeys{i}; %#ok<AGROW>
    end
end
if ~showAdaptive, algKeys = {}; end

fig = figure('Color', 'w', 'Name', ['Section 5.2 Summary - ' problem.name], ...
    'Visible', 'on', 'Position', [20 150 2400 500]);
xmax = section52_target_xmax(caseResult, solverOpts);
left = 0.035; right = 0.012; gap = 0.025;
bottom = 0.22; top = 0.10;
panelWidth = (1 - left - right - 4 * gap) / 5;
panelHeight = 1 - bottom - top;
axesList = gobjects(1, 5);
for p = 1:5
    axesList(p) = axes('Parent', fig, 'Position', ...
        [left + (p - 1) * (panelWidth + gap), bottom, panelWidth, panelHeight]); %#ok<LAXES>
    hold(axesList(p), 'on'); grid(axesList(p), 'on');
end

legendHandles = gobjects(1, numel(runKeys));
legendEntries = cell(1, numel(runKeys));
for i = 1:numel(runKeys)
    key = runKeys{i}; history = caseResult.histories.(key);
    style = section52_curve_style(caseResult, key);
    if isfield(history, 'time') && ~isempty(history.time)
        t = history.time(:);
    else
        t = (1:numel(history.obj)).';
    end
    legendHandles(i) = plot(axesList(1), t, history.obj(:), ...
        'LineWidth', 1.6, 'Color', style.Color, 'LineStyle', style.LineStyle, ...
        'Marker', style.Marker, 'MarkerIndices', section52_marker_indices(numel(t)), 'MarkerSize', 6);
    semilogy(axesList(2), t, max(history.residual(:), eps), ...
        'LineWidth', 1.6, 'Color', style.Color, 'LineStyle', style.LineStyle, ...
        'Marker', style.Marker, 'MarkerIndices', section52_marker_indices(numel(t)), 'MarkerSize', 6);
    plot(axesList(3), t, history.psnr(:), ...
        'LineWidth', 1.6, 'Color', style.Color, 'LineStyle', style.LineStyle, ...
        'Marker', style.Marker, 'MarkerIndices', section52_marker_indices(numel(t)), 'MarkerSize', 6);
    legendEntries{i} = section52_history_label(caseResult, key);
end

if isfield(solverOpts, 'objectiveScale')
    objectiveScale = solverOpts.objectiveScale;
else
    objectiveScale = 1 / problem.numSamples;
end
for i = 1:numel(algKeys)
    key = algKeys{i}; history = caseResult.histories.(key);
    style = section52_curve_style(caseResult, key);
    if isfield(history, 'time') && ~isempty(history.time)
        t = history.time(:);
    else
        t = (1:numel(history.beta)).';
    end
    beta = history.beta(:) / objectiveScale;
    valid = isfinite(t) & isfinite(beta);
    plot(axesList(4), t(valid), beta(valid), ...
        'LineWidth', 1.6, 'Color', style.Color, 'LineStyle', style.LineStyle, ...
        'Marker', style.Marker, 'MarkerIndices', section52_marker_indices(sum(valid)), 'MarkerSize', 6);
    if isfield(history, 'alpha')
        alpha = history.alpha(:); valid = isfinite(t) & isfinite(alpha);
        rawAlphaTime = t(valid);
        rawAlphaValue = alpha(valid);
        [alphaTime, alphaValue] = section52_binned_alpha_summary(rawAlphaTime, rawAlphaValue, 5);
        % Bin centers summarize the fluctuations well, but the final bin
        % center is earlier than the actual stopping time (for example,
        % about 47.5 s in a 50 s run).  Append the genuine last sample so
        % the plotted trace records the full runtime without extrapolation.
        if ~isempty(rawAlphaTime) && (isempty(alphaTime) || rawAlphaTime(end) > alphaTime(end))
            alphaTime = [alphaTime; rawAlphaTime(end)]; %#ok<AGROW>
            alphaValue = [alphaValue; rawAlphaValue(end)]; %#ok<AGROW>
        end
        plot(axesList(5), alphaTime, alphaValue, ...
            'LineWidth', 1.6, 'Color', style.Color, 'LineStyle', style.LineStyle, ...
            'Marker', style.Marker, 'MarkerIndices', section52_marker_indices(numel(alphaTime)), 'MarkerSize', 6);
    end
end

ylabels = {'objective value','constraint residual','PSNR (dB)', ...
    '\beta_k (original scale)','\alpha_k'};
for p = 1:5
    xlabel(axesList(p), 'time (sec.)'); ylabel(axesList(p), ylabels{p});
    xlim(axesList(p), [0 xmax]);
end
set(axesList(2), 'YScale', 'log');
if isempty(algKeys)
    text(axesList(4), 0.5, 0.5, 'adaptive curves disabled', 'Units', 'normalized', ...
        'HorizontalAlignment', 'center');
    text(axesList(5), 0.5, 0.5, 'adaptive curves disabled', 'Units', 'normalized', ...
        'HorizontalAlignment', 'center');
end

lgd = legend(axesList(3), legendHandles, legendEntries, ...
    'Interpreter', 'none', 'Orientation', 'horizontal', ...
    'NumColumns', numel(legendEntries), 'Units', 'normalized', 'Box', 'off');
drawnow;
lgdPos = lgd.Position;
lgdPos(1) = (1 - lgdPos(3)) / 2;
lgdPos(2) = 0.035;
set(lgd, 'Position', lgdPos);

% Show and retain the combined figure after all algorithms have finished.
% Saving the PNG below is an additional output, not a replacement for the
% interactive MATLAB figure window.
if doPlot
    set(fig, 'Visible', 'on');
    figure(fig);
    drawnow;
end

if doSave
    if ~exist(outDir, 'dir'), mkdir(outDir); end
    set(fig, 'PaperPositionMode', 'auto');
    print(fig, fullfile(outDir, [problem.name '_curves_adaptive.png']), '-dpng', '-r300');
end
if ~doPlot, close(fig); end
end

function plot_section52_case(problem, caseResult, outDir, doPlot, doSave)
fig = figure('Color', 'w', 'Name', ['Section 5.2 - ' problem.name], ...
    'Position', [80 180 max(1800, 280 * max(3, numel(fieldnames(caseResult.runs)) + 2)) 360]);
methodNames = {};
if isfield(caseResult, 'runs') && ~isempty(fieldnames(caseResult.runs))
    methodNames = section52_ordered_run_keys(caseResult, false);
end

numBasePanels = 2;
numMethodPanels = max(1, numel(methodNames));
numPanels = numBasePanels + numMethodPanels;
ncols = numPanels;
leftMargin = 0.012;
rightMargin = 0.012;
bottomMargin = 0.13;
topMargin = 0.12;
colGap = 0.010;
cellW = (1 - leftMargin - rightMargin - (ncols - 1) * colGap) / ncols;
cellH = 1 - topMargin - bottomMargin;

panelImages = cell(1, numPanels);
panelTopLabels = cell(1, numPanels);
panelBottomLabels = cell(1, numPanels);
panelImages{1} = problem.cleanImage;
panelTopLabels{1} = '';
panelBottomLabels{1} = 'Original';
panelImages{2} = problem.noisyImageClipped;
panelTopLabels{2} = sprintf('%.2f dB', psnr_gray(problem.noisyImageClipped, problem.cleanImage));
panelBottomLabels{2} = 'Noisy';

if ~isempty(methodNames)
    for i = 1:numel(methodNames)
        runKey = methodNames{i};
        xHat = caseResult.runs.(runKey).xHat;
        history = caseResult.runs.(runKey).history;
        if isfield(history, 'bestPSNR') && isfinite(history.bestPSNR)
            psnrShown = history.bestPSNR;
        else
            psnrShown = history.psnr(end);
        end
        panelImages{numBasePanels + i} = min(max(xHat, 0), 1);
        panelTopLabels{numBasePanels + i} = sprintf('%.2f dB', psnrShown);
        panelBottomLabels{numBasePanels + i} = section52_history_label(caseResult, runKey);
    end
else
    panelImages{3} = [];
    panelTopLabels{3} = '';
    panelBottomLabels{3} = 'No solverFcn provided';
end

for p = 1:numPanels
    axX = leftMargin + (p - 1) * (cellW + colGap);
    ax = axes('Parent', fig, 'Position', [axX, bottomMargin, cellW, cellH]); %#ok<LAXES>
    if isempty(panelImages{p})
        axis(ax, 'off');
        text(ax, 0.5, 0.5, panelBottomLabels{p}, 'FontSize', 11, ...
            'Units', 'normalized', 'HorizontalAlignment', 'center', ...
            'Interpreter', 'none');
    else
        imagesc(ax, panelImages{p});
        axis(ax, 'image', 'off');
        colormap(ax, gray);
        if ~isempty(panelTopLabels{p})
            text(ax, 0.5, 1.03, panelTopLabels{p}, 'Units', 'normalized', ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                'Interpreter', 'none', 'FontSize', 11, 'Clipping', 'off');
        end
        text(ax, 0.5, -0.05, panelBottomLabels{p}, 'Units', 'normalized', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
            'Interpreter', 'none', 'FontSize', 11, 'Clipping', 'off');
    end
end

if doSave
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    set(fig, 'PaperPositionMode', 'auto');
    print(fig, fullfile(outDir, [problem.name '_preview.png']), '-dpng', '-r300');
end
if ~doPlot
    close(fig);
end
end

function plot_section52_noise_summary(results, outDir, doPlot, doSave, figureName)
caseNames = fieldnames(results);
if isempty(caseNames)
    return;
end

fig = figure('Color', 'w', 'Name', 'Section 5.2 Stationary Noise Model', ...
    'Position', [80 60 1500 1020]);
nrows = numel(caseNames);
ncols = 4;
% MATLAB's default subplot margins are excessive for a compact image grid.
layout = struct('left', 0.070, 'right', 0.015, 'bottom', 0.035, 'top', 0.075, ...
    'columnGap', 0.018, 'rowGap', 0.030, 'nrows', nrows, 'ncols', ncols);

for i = 1:numel(caseNames)
    problem = results.(caseNames{i}).problem;

    ax1 = axes('Position', local_noise_panel_position(i, 1, layout));
    imagesc(problem.xTrue);
    axis image off;
    colormap(ax1, gray);
    if i == 1
        title('Ideal image ($x$)', 'Interpreter', 'latex');
    end
    text(ax1, -0.17, 0.5, local_case_name(problem), ...
        'Units', 'normalized', 'FontSize', 15, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'Rotation', 90, 'Clipping', 'off');

    ax2 = axes('Position', local_noise_panel_position(i, 2, layout));
    imagesc(problem.yTrue);
    axis image off;
    colormap(ax2, gray);
    if i == 1
        title('Texton ($b$), coefficient field ($y$)', 'Interpreter', 'latex');
    end
    local_add_texton_inset(ax2, problem.kernel);

    ax3 = axes('Position', local_noise_panel_position(i, 3, layout));
    imagesc(problem.noiseImage);
    axis image off;
    colormap(ax3, gray);
    if i == 1
        title('Stationary noise ($n$)', 'Interpreter', 'latex');
    end

    ax4 = axes('Position', local_noise_panel_position(i, 4, layout));
    imagesc(problem.noisyImageClipped);
    axis image off;
    colormap(ax4, gray);
    if i == 1
        title('Observation ($c$)', 'Interpreter', 'latex');
    end
end

if doSave
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    saveas(fig, fullfile(outDir, [figureName '.png']));
end
if ~doPlot
    close(fig);
end
end

function position = local_noise_panel_position(row, column, layout)
panelWidth = (1 - layout.left - layout.right - ...
    (layout.ncols - 1) * layout.columnGap) / layout.ncols;
panelHeight = (1 - layout.top - layout.bottom - ...
    (layout.nrows - 1) * layout.rowGap) / layout.nrows;
left = layout.left + (column - 1) * (panelWidth + layout.columnGap);
bottom = 1 - layout.top - row * panelHeight - (row - 1) * layout.rowGap;
position = [left, bottom, panelWidth, panelHeight];
end

function local_add_texton_inset(parentAx, kernel)
parentPos = get(parentAx, 'Position');
insetW = parentPos(3) * 0.42;
insetH = parentPos(4) * 0.22;
insetX = parentPos(1) + parentPos(3) * 0.02;
insetY = parentPos(2) + parentPos(4) * 0.75;
insetAx = axes('Position', [insetX insetY insetW insetH]); %#ok<LAXES>
imagesc(kernel);
axis image off;
colormap(insetAx, gray);
box(insetAx, 'on');
set(insetAx, 'XColor', 'w', 'YColor', 'w', 'LineWidth', 0.6);
end

function label = local_case_name(problem)
label = upper_first(problem.name);
end

function s = upper_first(s)
if isempty(s)
    return;
end
s = char(s);
s(1) = upper(s(1));
end

function plot_section52_curves(problem, caseResult, outDir, doPlot, doSave, solverOpts)
if ~isfield(caseResult, 'histories') && ~isfield(caseResult, 'history')
    return;
end
fig = figure('Color', 'w', 'Name', ['Section 5.2 Curves - ' problem.name], ...
    'Position', [80 120 1650 520]);
xmax = section52_target_xmax(caseResult, solverOpts);
legendHandles = [];
legendEntries = {};
axPos = [
    0.05 0.24 0.27 0.66;
    0.365 0.24 0.27 0.66;
    0.68 0.24 0.27 0.66
];

ax1 = subplot(1, 3, 1);
set(ax1, 'Units', 'normalized', 'Position', axPos(1, :));
hold on;
if isfield(caseResult, 'histories') && ~isempty(fieldnames(caseResult.histories))
    methodNames = section52_ordered_run_keys(caseResult, true);
    for i = 1:numel(methodNames)
        style = section52_curve_style(caseResult, methodNames{i});
        history = caseResult.histories.(methodNames{i});
        if ~isfield(history, 'time') || isempty(history.time)
            t = (1:numel(history.obj)).';
        else
            t = history.time(:);
        end
        hLine = plot(t, history.obj, ...
            'LineWidth', 1.6, ...
            'Color', style.Color, ...
            'LineStyle', style.LineStyle, ...
            'Marker', style.Marker, ...
            'MarkerIndices', section52_marker_indices(numel(t)), ...
            'MarkerSize', 6);
        legendHandles(end + 1) = hLine; %#ok<AGROW>
        legendEntries{end + 1} = section52_history_label(caseResult, methodNames{i}); %#ok<AGROW>
    end
else
    history = caseResult.history;
    if ~isfield(history, 'time') || isempty(history.time)
        t = (1:numel(history.obj)).';
    else
        t = history.time(:);
    end
    plot(t, history.obj, 'LineWidth', 1.6, 'Marker', 'o', 'MarkerIndices', section52_marker_indices(numel(t)), 'MarkerSize', 6);
end
grid on;
xlabel('time (sec.)');
ylabel('objective value');
xlim([0, xmax]);

ax2 = subplot(1, 3, 2);
set(ax2, 'Units', 'normalized', 'Position', axPos(2, :));
hold on;
if isfield(caseResult, 'histories') && ~isempty(fieldnames(caseResult.histories))
    methodNames = section52_ordered_run_keys(caseResult, true);
    for i = 1:numel(methodNames)
        style = section52_curve_style(caseResult, methodNames{i});
        history = caseResult.histories.(methodNames{i});
        if ~isfield(history, 'time') || isempty(history.time)
            t = (1:numel(history.residual)).';
        else
            t = history.time(:);
        end
        semilogy(t, max(history.residual, eps), ...
            'LineWidth', 1.6, ...
            'Color', style.Color, ...
            'LineStyle', style.LineStyle, ...
            'Marker', style.Marker, ...
            'MarkerIndices', section52_marker_indices(numel(t)), ...
            'MarkerSize', 6);
    end
else
    history = caseResult.history;
    if ~isfield(history, 'time') || isempty(history.time)
        t = (1:numel(history.residual)).';
    else
        t = history.time(:);
    end
    semilogy(t, max(history.residual, eps), 'LineWidth', 1.6, 'Marker', 'o', 'MarkerIndices', section52_marker_indices(numel(t)), 'MarkerSize', 6);
end
grid on;
xlabel('time (sec.)');
ylabel('constraint residual');
set(ax2, 'YScale', 'log');
xlim([0, xmax]);

ax3 = subplot(1, 3, 3);
set(ax3, 'Units', 'normalized', 'Position', axPos(3, :));
hold on;
if isfield(caseResult, 'histories') && ~isempty(fieldnames(caseResult.histories))
    methodNames = section52_ordered_run_keys(caseResult, true);
    for i = 1:numel(methodNames)
        history = caseResult.histories.(methodNames{i});
        if ~isfield(history, 'psnr') || isempty(history.psnr)
            continue;
        end
        style = section52_curve_style(caseResult, methodNames{i});
        if ~isfield(history, 'time') || isempty(history.time)
            t = (1:numel(history.psnr)).';
        else
            t = history.time(:);
        end
        plot(t, history.psnr, ...
            'LineWidth', 1.6, ...
            'Color', style.Color, ...
            'LineStyle', style.LineStyle, ...
            'Marker', style.Marker, ...
            'MarkerIndices', section52_marker_indices(numel(t)), ...
            'MarkerSize', 6);
    end
else
    history = caseResult.history;
    if isfield(history, 'psnr') && ~isempty(history.psnr)
        if ~isfield(history, 'time') || isempty(history.time)
            t = (1:numel(history.psnr)).';
        else
            t = history.time(:);
        end
        plot(t, history.psnr, 'LineWidth', 1.6, 'Marker', 'o', ...
            'MarkerIndices', section52_marker_indices(numel(t)), 'MarkerSize', 6);
    end
end
grid on;
xlabel('time (sec.)');
ylabel('PSNR (dB)');
xlim([0, xmax]);

if ~isempty(legendHandles)
    lgd = legend(ax3, legendHandles, legendEntries, ...
        'Interpreter', 'none', ...
        'Orientation', 'horizontal', ...
        'NumColumns', max(1, numel(caseResult.methods)), ...
        'Units', 'normalized');
    drawnow;
    lgdPos = lgd.Position;
    lgdPos(1) = (1 - lgdPos(3)) / 2;
    lgdPos(2) = 0.04;
    set(lgd, 'Position', lgdPos);
    set(lgd, 'Box', 'off');
end

if doSave
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    saveas(fig, fullfile(outDir, [problem.name '_curves.png']));
end
if ~doPlot
    close(fig);
end
end

function plot_section52_adaptive_parameters(problem, caseResult, outDir, doPlot, doSave, solverOpts)
% Plot the adaptive penalty and primal step only for the proposed Alg. 1 variants.
if ~isfield(caseResult, 'histories') || isempty(fieldnames(caseResult.histories))
    return;
end

runKeys = section52_ordered_run_keys(caseResult, true);
algKeys = {};
for i = 1:numel(runKeys)
    nameParts = split(string(runKeys{i}), "_b");
    methodName = char(nameParts(1));
    if any(strcmpi(methodName, {'sarah', 'svrg', 'saga'}))
        algKeys{end + 1} = runKeys{i}; %#ok<AGROW>
    end
end
if isempty(algKeys)
    return;
end

fig = figure('Color', 'w', 'Name', ['Section 5.2 Adaptive Parameters - ' problem.name], ...
    'Position', [80 120 1350 520]);
xmax = section52_target_xmax(caseResult, solverOpts);
if isfield(solverOpts, 'objectiveScale')
    objectiveScale = solverOpts.objectiveScale;
else
    objectiveScale = 1 / problem.numSamples;
end
legendHandles = [];
legendEntries = {};

ax1 = subplot(1, 2, 1);
set(ax1, 'Units', 'normalized', 'Position', [0.06 0.24 0.42 0.66]);
hold on;
ax2 = subplot(1, 2, 2);
set(ax2, 'Units', 'normalized', 'Position', [0.53 0.24 0.42 0.66]);
hold on;

for i = 1:numel(algKeys)
    runKey = algKeys{i};
    history = caseResult.histories.(runKey);
    style = section52_curve_style(caseResult, runKey);
    if ~isfield(history, 'time') || isempty(history.time)
        t = (1:numel(history.beta)).';
    else
        t = history.time(:);
    end

    % The solver stores beta for the normalized objective.  Display the
    % original problem scale used to state the experimental parameters.
    beta = history.beta(:) / objectiveScale;
    betaValid = isfinite(t) & isfinite(beta);
    axes(ax1);
    hLine = plot(t(betaValid), beta(betaValid), ...
        'LineWidth', 1.6, 'Color', style.Color, 'LineStyle', style.LineStyle, ...
        'Marker', style.Marker, ...
        'MarkerIndices', section52_marker_indices(sum(betaValid)), 'MarkerSize', 6);
    legendHandles(end + 1) = hLine; %#ok<AGROW>
    legendEntries{end + 1} = section52_history_label(caseResult, runKey); %#ok<AGROW>

    if isfield(history, 'alpha')
        alpha = history.alpha(:);
        alphaValid = isfinite(t) & isfinite(alpha);
        alphaTime = t(alphaValid);
        alpha = alpha(alphaValid);
        % A per-iteration trace is unreadable for high-frequency stochastic
        % steps.  The paper figure shows only a robust 5-second median
        % trend; raw fluctuations can be reported separately if needed.
        [alphaTime, alpha, ~, ~] = ...
            section52_binned_alpha_summary(alphaTime, alpha, 5);
        axes(ax2);
        plot(alphaTime, alpha, ...
            'LineWidth', 1.6, 'Color', style.Color, 'LineStyle', style.LineStyle, ...
            'Marker', style.Marker, ...
            'MarkerIndices', section52_marker_indices(numel(alphaTime)), 'MarkerSize', 6);
    end
end

axes(ax1);
grid on;
xlabel('time (sec.)');
ylabel('\beta_k (original scale)');
xlim([0, xmax]);

axes(ax2);
grid on;
xlabel('time (sec.)');
ylabel('\alpha_k');
xlim([0, xmax]);

if ~isempty(legendHandles)
    lgd = legend(ax2, legendHandles, legendEntries, 'Interpreter', 'none', ...
        'Orientation', 'horizontal', 'NumColumns', numel(legendEntries), ...
        'Units', 'normalized', 'Box', 'off');
    drawnow;
    lgdPos = lgd.Position;
    lgdPos(1) = (1 - lgdPos(3)) / 2;
    lgdPos(2) = 0.04;
    set(lgd, 'Position', lgdPos);
end

if doSave
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    saveas(fig, fullfile(outDir, [problem.name '_adaptive_parameters.png']));
end
if ~doPlot
    close(fig);
end
end

function plot_section52_decomposition(problem, caseResult, outDir, doPlot, doSave)
if ~isfield(caseResult, 'runs') || isempty(fieldnames(caseResult.runs))
    return;
end

runKeys = section52_ordered_run_keys(caseResult, false);
fig = figure('Color', 'w', 'Name', ['Section 5.2 Decomposition - ' problem.name]);
nrows = numel(runKeys);
ncols = 3;

for i = 1:numel(runKeys)
    runKey = runKeys{i};
    runData = caseResult.runs.(runKey);
    label = section52_history_label(caseResult, runKey);

    subplot(nrows, ncols, (i - 1) * ncols + 1);
    imagesc(min(max(runData.xHat, 0), 1));
    axis image off;
    colormap gray;
    if isfield(runData.history, 'bestPSNR') && isfinite(runData.history.bestPSNR)
        ttl = sprintf('%s\nx_hat, PSNR %.2f dB', label, runData.history.bestPSNR);
    else
        ttl = sprintf('%s\nx_hat', label);
    end
    title(ttl, 'Interpreter', 'none');

    subplot(nrows, ncols, (i - 1) * ncols + 2);
    imagesc(runData.ByHat);
    axis image off;
    colormap gray;
    title('B y_hat');

    subplot(nrows, ncols, (i - 1) * ncols + 3);
    imagesc(runData.residualImage);
    axis image off;
    colormap gray;
    title(sprintf('residual, ||r||=%.2e', norm(runData.residualImage(:))));
end

if doSave
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    saveas(fig, fullfile(outDir, [problem.name '_decomposition.png']));
end
if ~doPlot
    close(fig);
end
end

function xmax = section52_target_xmax(caseResult, solverOpts)
xmax = 1;
if isfield(caseResult, 'histories') && ~isempty(fieldnames(caseResult.histories))
    methodNames = fieldnames(caseResult.histories);
    for i = 1:numel(methodNames)
        history = caseResult.histories.(methodNames{i});
        if isfield(history, 'time') && ~isempty(history.time)
            xmax = max(xmax, history.time(end));
        end
    end
elseif isfield(caseResult, 'history') && isfield(caseResult.history, 'time') && ~isempty(caseResult.history.time)
    xmax = max(xmax, caseResult.history.time(end));
end
if nargin >= 2 && isfield(solverOpts, 'maxTime') && isfinite(solverOpts.maxTime) ...
        && ~(isfield(solverOpts, 'psnrTarget') && isfinite(solverOpts.psnrTarget))
    xmax = max(xmax, solverOpts.maxTime);
end
end

function plot_section52_objective_breakdown(problem, caseResult, outDir, doPlot, doSave, solverOpts)
if ~isfield(caseResult, 'histories') || isempty(fieldnames(caseResult.histories))
    return;
end

fig = figure('Color', 'w', 'Name', ['Section 5.2 Objective Breakdown - ' problem.name]);
xmax = section52_target_xmax(caseResult, solverOpts);
legendHandles = [];
legendEntries = {};
axPos = [
    0.08 0.24 0.26 0.68;
    0.39 0.24 0.26 0.68;
    0.70 0.24 0.26 0.68
];

methodNames = section52_ordered_run_keys(caseResult, true);

ax1 = subplot(1, 3, 1);
set(ax1, 'Position', axPos(1, :));
hold on;
for i = 1:numel(methodNames)
    style = section52_curve_style(caseResult, methodNames{i});
    history = caseResult.histories.(methodNames{i});
    t = history.time(:);
    hLine = plot(t, history.fval, ...
        'LineWidth', 1.6, ...
        'Color', style.Color, ...
        'LineStyle', style.LineStyle, ...
        'Marker', style.Marker, ...
        'MarkerIndices', section52_marker_indices(numel(t)), ...
        'MarkerSize', 6);
    legendHandles(end + 1) = hLine; %#ok<AGROW>
    legendEntries{end + 1} = section52_history_label(caseResult, methodNames{i}); %#ok<AGROW>
end
grid on;
xlabel('time (sec.)');
ylabel('f(x)');
xlim([0, xmax]);

ax2 = subplot(1, 3, 2);
set(ax2, 'Position', axPos(2, :));
hold on;
for i = 1:numel(methodNames)
    style = section52_curve_style(caseResult, methodNames{i});
    history = caseResult.histories.(methodNames{i});
    t = history.time(:);
    plot(t, history.gval, ...
        'LineWidth', 1.6, ...
        'Color', style.Color, ...
        'LineStyle', style.LineStyle, ...
        'Marker', style.Marker, ...
        'MarkerIndices', section52_marker_indices(numel(t)), ...
        'MarkerSize', 6);
end
grid on;
xlabel('time (sec.)');
ylabel('\lambda ||y||_1');
xlim([0, xmax]);

ax3 = subplot(1, 3, 3);
set(ax3, 'Position', axPos(3, :));
hold on;
for i = 1:numel(methodNames)
    style = section52_curve_style(caseResult, methodNames{i});
    history = caseResult.histories.(methodNames{i});
    t = history.time(:);
    plot(t, history.obj, ...
        'LineWidth', 1.6, ...
        'Color', style.Color, ...
        'LineStyle', style.LineStyle, ...
        'Marker', style.Marker, ...
        'MarkerIndices', section52_marker_indices(numel(t)), ...
        'MarkerSize', 6);
end
grid on;
xlabel('time (sec.)');
ylabel('f(x) + \lambda ||y||_1');
xlim([0, xmax]);

if ~isempty(legendHandles)
    lgd = legend(ax3, legendHandles, legendEntries, ...
        'Interpreter', 'none', ...
        'Orientation', 'horizontal', ...
        'NumColumns', max(1, numel(caseResult.methods)), ...
        'Units', 'normalized');
    set(lgd, 'Position', [0.10 0.03 0.80 0.11]);
end

if doSave
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    saveas(fig, fullfile(outDir, [problem.name '_objective_breakdown.png']));
end
if ~doPlot
    close(fig);
end
end

function runKey = section52_run_key(methodName, batchFraction)
if isnan(batchFraction)
    runKey = methodName;
else
    runKey = sprintf('%s_b%03d', methodName, round(100 * batchFraction));
end
end

function label = section52_legend_label(methodName, batchFraction, batchSize, numSamples) %#ok<INUSD>
switch lower(methodName)
    case 'sarah'
        label = 'Alg.1-sarah';
    case 'svrg'
        label = 'Alg.1-svrg';
    case 'saga'
        label = 'Alg.1-saga';
    case {'bsarah', 'bsarahsadmm'}
        label = 'SARAH-ADMM';
    case {'svrgadmm', 'svrg_admm'}
        label = 'SVRG-ADMM';
    case {'sadmm_timevarying', 'sadmm-timevarying', 'sadmm_tv'}
        label = 'SADMM';
    otherwise
        label = methodName;
end
end

function runKeys = section52_ordered_run_keys(caseResult, useHistories)
if useHistories
    if ~isfield(caseResult, 'histories') || isempty(fieldnames(caseResult.histories))
        runKeys = {};
        return;
    end
    available = fieldnames(caseResult.histories);
else
    if ~isfield(caseResult, 'runs') || isempty(fieldnames(caseResult.runs))
        runKeys = {};
        return;
    end
    available = fieldnames(caseResult.runs);
end

runKeys = {};
configuredMethods = caseResult.methods;
methodRanks = zeros(1, numel(configuredMethods));
for m = 1:numel(configuredMethods)
    methodRanks(m) = section52_display_method_rank(configuredMethods{m});
end
[~, displayOrder] = sortrows([methodRanks(:), (1:numel(configuredMethods)).']);
displayMethods = configuredMethods(displayOrder);
if isempty(caseResult.batchFractions)
    batchFractions = NaN;
else
    batchFractions = caseResult.batchFractions(:).';
end

for b = 1:numel(batchFractions)
    frac = batchFractions(b);
    for m = 1:numel(displayMethods)
        key = section52_run_key(lower(displayMethods{m}), frac);
        if any(strcmp(available, key))
            runKeys{end + 1} = key; %#ok<AGROW>
        end
    end
end

extras = setdiff(available, runKeys, 'stable');
runKeys = [runKeys, extras];
end

function rank = section52_display_method_rank(methodName)
switch lower(methodName)
    case {'sadmm_timevarying', 'sadmm-timevarying', 'sadmm_tv'}
        rank = 1;
    case {'svrgadmm', 'svrg_admm'}
        rank = 2;
    case {'bsarah', 'bsarahsadmm'}
        rank = 3;
    case 'svrg'
        rank = 4;
    case 'saga'
        rank = 5;
    case 'sarah'
        rank = 6;
    otherwise
        rank = 100;
end
end

function label = section52_history_label(caseResult, runKey)
if isfield(caseResult, 'legendLabels') && isfield(caseResult.legendLabels, runKey)
    label = caseResult.legendLabels.(runKey);
else
    label = runKey;
end
end

function style = section52_curve_style(caseResult, runKey)
style = struct('Color', [0 0.4470 0.7410], 'LineStyle', '-', 'Marker', 'o');

parts = split(string(runKey), "_b");
methodName = char(parts(1));
batchToken = "";
if numel(parts) > 1
    batchToken = parts(2);
end

switch lower(methodName)
    case 'sarah'
        style.Color = [1 0 0];
        style.Marker = '^';
        style.LineStyle = '-';
    case 'svrg'
        style.Color = [0 0 1];
        style.Marker = 's';
        style.LineStyle = '-';
    case 'saga'
        style.Color = [1 0 1];
        style.Marker = 'o';
        style.LineStyle = '-';
    case {'bsarah', 'bsarahsadmm'}
        style.Color = [0 0.8000 0.2000];
        style.Marker = 'd';
        style.LineStyle = '--';
    case {'svrgadmm', 'svrg_admm'}
        style.Color = [0.8500 0.3250 0.0980];
        style.Marker = '*';
        style.LineStyle = '--';
    case {'sadmm_timevarying', 'sadmm-timevarying', 'sadmm_tv'}
        style.Color = [0 0.8500 0.9000];
        style.Marker = 'v';
        style.LineStyle = '--';
    otherwise
        style.Color = [0.3000 0.3000 0.3000];
        style.Marker = 'x';
        style.LineStyle = '-';
end

if isfield(caseResult, 'batchFractions') && numel(caseResult.batchFractions) > 1
    switch char(batchToken)
        case '004'
            style.LineStyle = '--';
        case '005'
            style.LineStyle = '-.';
    end
end
end

function idx = section52_marker_indices(n)
if n <= 12
    idx = 1:n;
else
    idx = unique(round(linspace(1, n, 10)));
end
end

function [timeCenter, medianValue, lowerValue, upperValue] = ...
    section52_binned_alpha_summary(t, value, binWidth)
% Robustly summarize a dense stochastic trace in elapsed-time bins.
t = t(:);
value = value(:);
if isempty(t)
    timeCenter = zeros(0, 1);
    medianValue = zeros(0, 1);
    lowerValue = zeros(0, 1);
    upperValue = zeros(0, 1);
    return;
end

timeBin = floor((t - t(1)) / binWidth);
bins = unique(timeBin, 'stable');
nBins = numel(bins);
timeCenter = zeros(nBins, 1);
medianValue = zeros(nBins, 1);
lowerValue = zeros(nBins, 1);
upperValue = zeros(nBins, 1);
for i = 1:nBins
    inBin = timeBin == bins(i);
    values = value(inBin);
    timeCenter(i) = median(t(inBin));
    medianValue(i) = median(values);
    lowerValue(i) = prctile(values, 10);
    upperValue(i) = prctile(values, 90);
end
end
