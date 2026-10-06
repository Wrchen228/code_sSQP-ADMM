clear;
clc;
close all;

%% 1. 加载数据集

dataset_files = { ...
    'datasets/Adult.mat', ...
    'datasets/IJCNN.mat', ...
    'datasets/covtype.mat', ...
    'datasets/SUSY.mat' ...
    };

for dataset_index = 1:numel(dataset_files)
dataset_file = dataset_files{dataset_index};
fprintf('\n\n############################################################\n');
fprintf('开始运行数据集 %d / %d：%s\n', ...
    dataset_index, numel(dataset_files), dataset_file);
fprintf('############################################################\n');
[X, y] = load_binary_dataset(dataset_file);

num_samples = size(X, 1);
dim = size(X, 2);
%% 2. 按数据集选择独立参数
[~, dataset_name, ~] = fileparts(dataset_file);
switch lower(dataset_name)
    case 'adult'
        p = Set_adult_Params(struct());
    case 'susy'
        p = Set_susy_Params(struct());
    case 'ijcnn'
        p = Set_ijcnn_Params(struct());
    case 'covtype'
        p = Set_covtype_binary_scaled_Params(struct());
    otherwise
        p = Set_tongyi(struct());
        p.max_time = 100;
        p.num_inner_iters = 10;
        p.batch_size = 128;
end

if ~isfield(p, 'parameter_trace_max_iters')
    p.parameter_trace_max_iters = 2000;
end
if ~isfield(p, 'parameter_trace_change_tol')
    p.parameter_trace_change_tol = 1e-5;
end
if ~isfield(p, 'parameter_stability_window')
    p.parameter_stability_window = 50;
end
if ~isfield(p, 'refresh_prob') || ~isscalar(p.refresh_prob) || ...
        p.refresh_prob <= 0 || p.refresh_prob > 1
    error(['Each data-set parameter package must define ', ...
        'p.refresh_prob in (0, 1].']);
end

max_time = p.max_time;
num_inner_iters = p.num_inner_iters;
batch_size = p.batch_size;
p.outer_record_interval = num_inner_iters;
mu = 1e-5;            % 正则化系数
A = eye(dim);
Num_Runs = 5;         % 独立重复实验次数：逐次报告，不取平均
% Each independent run is plotted separately after all runs finish.

% 初始化存储器（t：时间，f：测试集误差，obj：训练集目标值）
all_t_bsarah = cell(Num_Runs, 1); all_f_bsarah = cell(Num_Runs, 1); all_obj_bsarah = cell(Num_Runs, 1);
all_t_bsvrg  = cell(Num_Runs, 1); all_f_bsvrg  = cell(Num_Runs, 1); all_obj_bsvrg  = cell(Num_Runs, 1);
all_t_sadmm  = cell(Num_Runs, 1); all_f_sadmm  = cell(Num_Runs, 1); all_obj_sadmm  = cell(Num_Runs, 1);
all_t_ssvrg  = cell(Num_Runs, 1); all_f_ssvrg  = cell(Num_Runs, 1); all_obj_ssvrg  = cell(Num_Runs, 1);
all_t_ssaga  = cell(Num_Runs, 1); all_f_ssaga  = cell(Num_Runs, 1); all_obj_ssaga  = cell(Num_Runs, 1);
all_t_ssarah = cell(Num_Runs, 1); all_f_ssarah = cell(Num_Runs, 1); all_obj_ssarah = cell(Num_Runs, 1);
all_r_bsarah = cell(Num_Runs, 1); all_r_bsvrg = cell(Num_Runs, 1);
all_r_sadmm = cell(Num_Runs, 1); all_r_ssvrg = cell(Num_Runs, 1);
all_r_ssaga = cell(Num_Runs, 1); all_r_ssarah = cell(Num_Runs, 1);
all_early_k_ssvrg  = cell(Num_Runs, 1); all_early_beta_ssvrg  = cell(Num_Runs, 1); all_early_alpha_ssvrg  = cell(Num_Runs, 1);
all_early_k_ssaga  = cell(Num_Runs, 1); all_early_beta_ssaga  = cell(Num_Runs, 1); all_early_alpha_ssaga  = cell(Num_Runs, 1);
all_early_k_ssarah = cell(Num_Runs, 1); all_early_beta_ssarah = cell(Num_Runs, 1); all_early_alpha_ssarah = cell(Num_Runs, 1);

res_final_bsarah = zeros(Num_Runs, 1);
res_final_bsvrg  = zeros(Num_Runs, 1);
res_final_sadmm  = zeros(Num_Runs, 1);
res_final_ssvrg  = zeros(Num_Runs, 1);
res_final_ssaga  = zeros(Num_Runs, 1);
res_final_ssarah = zeros(Num_Runs, 1);
best_res_bsarah = zeros(Num_Runs, 1); best_obj_bsarah = zeros(Num_Runs, 1); best_time_bsarah = zeros(Num_Runs, 1);
best_res_bsvrg  = zeros(Num_Runs, 1); best_obj_bsvrg  = zeros(Num_Runs, 1); best_time_bsvrg  = zeros(Num_Runs, 1);
best_res_sadmm  = zeros(Num_Runs, 1); best_obj_sadmm  = zeros(Num_Runs, 1); best_time_sadmm  = zeros(Num_Runs, 1);
best_res_ssvrg  = zeros(Num_Runs, 1); best_obj_ssvrg  = zeros(Num_Runs, 1); best_time_ssvrg  = zeros(Num_Runs, 1);
best_res_ssaga  = zeros(Num_Runs, 1); best_obj_ssaga  = zeros(Num_Runs, 1); best_time_ssaga  = zeros(Num_Runs, 1);
best_res_ssarah = zeros(Num_Runs, 1); best_obj_ssarah = zeros(Num_Runs, 1); best_time_ssarah = zeros(Num_Runs, 1);
class_error = zeros(Num_Runs, 6);
balanced_accuracy = zeros(Num_Runs, 6);

%% 3. 外层循环：按论文采用 50/50 随机划分并独立运行
for run = 1:Num_Runs
    fprintf('====== 正在执行第 %d / %d 次独立数值实验（限时 %d 秒）======\n', run, Num_Runs, max_time);
    
    y = y(:);  % 强制转换为列向量
    rng(run + 2026);
    shuffled_indices = randperm(num_samples);
    train_size = round(0.5 * num_samples);
    idx_train = shuffled_indices(1:train_size);
    idx_test = shuffled_indices(train_size+1:end);

   Data_train = [X(idx_train, :), y(idx_train)];
   Data_test  = [X(idx_test,  :), y(idx_test)];

    % The three loopless estimators use exactly this shared probability:
    % B-SARAH-SADMM, SSQP-ADMM-svrg, and SSQP-ADMM-sarah.
    % SAGA has no full-gradient refresh; B-SVRG uses its paper's fixed epoch.
    fprintf(['训练样本数 N_train = %d，batch size = %d，三个 loopless 算法的共享刷新概率 ', ...
        'p = %.6e（平均每 %.0f 次迭代刷新）\n'], ...
        size(Data_train, 1), batch_size, p.refresh_prob, 1 / p.refresh_prob);
    
    x0 = zeros(dim, 1);
    y0 = zeros(dim, 1);
    % --- 运行 6 种不同的算法（接收指标轨迹与最终模型）---
    algorithm_seed = run + 22026;
    rng(algorithm_seed);
    [test_loss_bsarah, t_bsarah, r_bsarah, train_obj_bsarah, x_bsarah] = ...
        BSARAHSADMM(Data_train, Data_test, x0, y0, mu, A, max_time, ...
        batch_size, p.record_interval, p.beta, p.refresh_prob);
    rng(algorithm_seed);
    [test_loss_bsvrg, t_bsvrg, r_bsvrg, train_obj_bsvrg, x_bsvrg] = ...
        BSVRGADMM(Data_train, Data_test, x0, y0, mu, A, max_time, ...
        batch_size, num_inner_iters, p.record_interval, p.beta);
    rng(algorithm_seed);
    [test_loss_sadmm, t_sadmm, r_sadmm, train_obj_sadmm, x_sadmm] = ...
        SADMM_timevarying(Data_train, Data_test, x0, y0, mu, A, max_time, ...
        p.record_interval, p.beta);
    rng(algorithm_seed);
    [test_loss_ssvrg, t_ssvrg, r_ssvrg, train_obj_ssvrg, x_ssvrg, ...
        ~, ~, ~, ~, early_beta_ssvrg, ...
        early_alpha_ssvrg, early_k_ssvrg] = ...
        SQPADMM_svrg(Data_train, Data_test, x0, y0, mu, A, max_time, batch_size, p);
    rng(algorithm_seed);
    [test_loss_ssaga, t_ssaga, r_ssaga, train_obj_ssaga, x_ssaga, ...
        ~, ~, ~, ~, early_beta_ssaga, ...
        early_alpha_ssaga, early_k_ssaga] = ...
        SQPADMM_saga(Data_train, Data_test, x0, y0, mu, A, max_time, batch_size, p);
    rng(algorithm_seed);
    [test_loss_ssarah, t_ssarah, r_ssarah, train_obj_ssarah, x_ssarah, ...
        ~, ~, ~, ~, early_beta_ssarah, ...
        early_alpha_ssarah, early_k_ssarah] = ...
        SQPADMM_sarah(Data_train, Data_test, x0, y0, mu, A, max_time, batch_size, p);
    
    % --- 存储离散的时间、测试误差和训练目标轨迹 ---
    all_t_bsarah{run} = t_bsarah(:); all_f_bsarah{run} = test_loss_bsarah(:); all_obj_bsarah{run} = train_obj_bsarah(:);
    all_t_bsvrg{run} = t_bsvrg(:);   all_f_bsvrg{run} = test_loss_bsvrg(:);   all_obj_bsvrg{run} = train_obj_bsvrg(:);
    all_t_sadmm{run} = t_sadmm(:);   all_f_sadmm{run} = test_loss_sadmm(:);   all_obj_sadmm{run} = train_obj_sadmm(:);
    all_t_ssvrg{run} = t_ssvrg(:);   all_f_ssvrg{run} = test_loss_ssvrg(:);   all_obj_ssvrg{run} = train_obj_ssvrg(:);
    all_t_ssaga{run} = t_ssaga(:);   all_f_ssaga{run} = test_loss_ssaga(:);   all_obj_ssaga{run} = train_obj_ssaga(:);
    all_t_ssarah{run} = t_ssarah(:); all_f_ssarah{run} = test_loss_ssarah(:); all_obj_ssarah{run} = train_obj_ssarah(:);
    all_r_bsarah{run} = r_bsarah(:); all_r_bsvrg{run} = r_bsvrg(:);
    all_r_sadmm{run} = r_sadmm(:); all_r_ssvrg{run} = r_ssvrg(:);
    all_r_ssaga{run} = r_ssaga(:); all_r_ssarah{run} = r_ssarah(:);
    all_early_k_ssvrg{run} = early_k_ssvrg(:);     all_early_beta_ssvrg{run} = early_beta_ssvrg(:);     all_early_alpha_ssvrg{run} = early_alpha_ssvrg(:);
    all_early_k_ssaga{run} = early_k_ssaga(:);     all_early_beta_ssaga{run} = early_beta_ssaga(:);     all_early_alpha_ssaga{run} = early_alpha_ssaga(:);
    all_early_k_ssarah{run} = early_k_ssarah(:);   all_early_beta_ssarah{run} = early_beta_ssarah(:);   all_early_alpha_ssarah{run} = early_alpha_ssarah(:);
    
    res_final_bsarah(run) = r_bsarah(end);
    res_final_bsvrg(run)  = r_bsvrg(end);
    res_final_sadmm(run)  = r_sadmm(end);
    res_final_ssvrg(run)  = r_ssvrg(end);
    res_final_ssaga(run)  = r_ssaga(end);
    res_final_ssarah(run) = r_ssarah(end);

    [best_res_bsarah(run), best_idx] = minimum_residual_and_index(r_bsarah);
    best_obj_bsarah(run) = recorded_value_at_index(train_obj_bsarah, best_idx); best_time_bsarah(run) = recorded_value_at_index(t_bsarah, best_idx);
    [best_res_bsvrg(run), best_idx] = minimum_residual_and_index(r_bsvrg);
    best_obj_bsvrg(run) = recorded_value_at_index(train_obj_bsvrg, best_idx); best_time_bsvrg(run) = recorded_value_at_index(t_bsvrg, best_idx);
    [best_res_sadmm(run), best_idx] = minimum_residual_and_index(r_sadmm);
    best_obj_sadmm(run) = recorded_value_at_index(train_obj_sadmm, best_idx); best_time_sadmm(run) = recorded_value_at_index(t_sadmm, best_idx);
    [best_res_ssvrg(run), best_idx] = minimum_residual_and_index(r_ssvrg);
    best_obj_ssvrg(run) = recorded_value_at_index(train_obj_ssvrg, best_idx); best_time_ssvrg(run) = recorded_value_at_index(t_ssvrg, best_idx);
    [best_res_ssaga(run), best_idx] = minimum_residual_and_index(r_ssaga);
    best_obj_ssaga(run) = recorded_value_at_index(train_obj_ssaga, best_idx); best_time_ssaga(run) = recorded_value_at_index(t_ssaga, best_idx);
    [best_res_ssarah(run), best_idx] = minimum_residual_and_index(r_ssarah);
    best_obj_ssarah(run) = recorded_value_at_index(train_obj_ssarah, best_idx); best_time_ssarah(run) = recorded_value_at_index(t_ssarah, best_idx);

    final_models = {x_bsarah, x_bsvrg, x_sadmm, x_ssvrg, x_ssaga, x_ssarah};
    for algorithm_id = 1:numel(final_models)
        [class_error(run, algorithm_id), balanced_accuracy(run, algorithm_id)] = ...
            classification_metrics(Data_test, final_models{algorithm_id});
    end
end

%% 4. 在统一时间轴上进行插值和对齐
for plot_run = 1:Num_Runs
fprintf('正在生成第 %d / %d 次独立运行的图像（不取平均）...\n', plot_run, Num_Runs);
fprintf('正在将离散轨迹对齐到统一的 %g 秒时间轴...\n', max_time);
grid_pts = 200; 
mean_t = linspace(0, max_time, grid_pts)'; 

% 对齐测试集 Loss
mean_loss_bsarah = interp_and_average(all_t_bsarah(plot_run), all_f_bsarah(plot_run), mean_t);
mean_loss_bsvrg  = interp_and_average(all_t_bsvrg(plot_run), all_f_bsvrg(plot_run), mean_t);
mean_loss_sadmm  = interp_and_average(all_t_sadmm(plot_run), all_f_sadmm(plot_run), mean_t);
mean_loss_ssvrg  = interp_and_average(all_t_ssvrg(plot_run), all_f_ssvrg(plot_run), mean_t);
mean_loss_ssaga  = interp_and_average(all_t_ssaga(plot_run), all_f_ssaga(plot_run), mean_t);
mean_loss_ssarah = interp_and_average(all_t_ssarah(plot_run), all_f_ssarah(plot_run), mean_t);

% Align residuals from the first actual recording time.  The point t=0 is
% exactly feasible because x0=y0=0; it must not be log-plotted or used to
% interpolate the transient residual curve.
residual_time = linspace(p.record_interval, max_time, grid_pts - 1)';
mean_res_bsarah = interp_residual_and_average(all_t_bsarah(plot_run), all_r_bsarah(plot_run), residual_time);
mean_res_bsvrg  = interp_residual_and_average(all_t_bsvrg(plot_run), all_r_bsvrg(plot_run), residual_time);
mean_res_sadmm  = interp_residual_and_average(all_t_sadmm(plot_run), all_r_sadmm(plot_run), residual_time);
mean_res_ssvrg  = interp_residual_and_average(all_t_ssvrg(plot_run), all_r_ssvrg(plot_run), residual_time);
mean_res_ssaga  = interp_residual_and_average(all_t_ssaga(plot_run), all_r_ssaga(plot_run), residual_time);
mean_res_ssarah = interp_residual_and_average(all_t_ssarah(plot_run), all_r_ssarah(plot_run), residual_time);

% 对齐训练集 Objective 值
mean_obj_bsarah = interp_and_average(all_t_bsarah(plot_run), all_obj_bsarah(plot_run), mean_t);
mean_obj_bsvrg  = interp_and_average(all_t_bsvrg(plot_run), all_obj_bsvrg(plot_run), mean_t);
mean_obj_sadmm  = interp_and_average(all_t_sadmm(plot_run), all_obj_sadmm(plot_run), mean_t);
mean_obj_ssvrg  = interp_and_average(all_t_ssvrg(plot_run), all_obj_ssvrg(plot_run), mean_t);
mean_obj_ssaga  = interp_and_average(all_t_ssaga(plot_run), all_obj_ssaga(plot_run), mean_t);
mean_obj_ssarah = interp_and_average(all_t_ssarah(plot_run), all_obj_ssarah(plot_run), mean_t);

% 对齐逐次随机更新的参数轨迹
[early_k_ssvrg, mean_early_beta_ssvrg] = average_by_iteration(all_early_k_ssvrg(plot_run), all_early_beta_ssvrg(plot_run));
[early_k_ssaga, mean_early_beta_ssaga] = average_by_iteration(all_early_k_ssaga(plot_run), all_early_beta_ssaga(plot_run));
[early_k_ssarah, mean_early_beta_ssarah] = average_by_iteration(all_early_k_ssarah(plot_run), all_early_beta_ssarah(plot_run));
[early_alpha_k_ssvrg, mean_early_alpha_ssvrg] = average_by_iteration(all_early_k_ssvrg(plot_run), all_early_alpha_ssvrg(plot_run));
[early_alpha_k_ssaga, mean_early_alpha_ssaga] = average_by_iteration(all_early_k_ssaga(plot_run), all_early_alpha_ssaga(plot_run));
[early_alpha_k_ssarah, mean_early_alpha_ssarah] = average_by_iteration(all_early_k_ssarah(plot_run), all_early_alpha_ssarah(plot_run));

% Plot until each estimator satisfies the beta stability test on a
% consecutive window. Alpha is displayed but does not affect truncation.
parameter_plot_end = max([ ...
    first_beta_window_stability_k(early_k_ssvrg, mean_early_beta_ssvrg, ...
        p.parameter_trace_change_tol, p.parameter_stability_window), ...
    first_beta_window_stability_k(early_k_ssaga, mean_early_beta_ssaga, ...
        p.parameter_trace_change_tol, p.parameter_stability_window), ...
    first_beta_window_stability_k(early_k_ssarah, mean_early_beta_ssarah, ...
        p.parameter_trace_change_tol, p.parameter_stability_window)]);
[plot_k_ssvrg, plot_beta_ssvrg] = trim_trace(early_k_ssvrg, mean_early_beta_ssvrg, parameter_plot_end);
[plot_k_ssaga, plot_beta_ssaga] = trim_trace(early_k_ssaga, mean_early_beta_ssaga, parameter_plot_end);
[plot_k_ssarah, plot_beta_ssarah] = trim_trace(early_k_ssarah, mean_early_beta_ssarah, parameter_plot_end);
[plot_alpha_k_ssvrg, plot_alpha_ssvrg] = trim_trace(early_alpha_k_ssvrg, mean_early_alpha_ssvrg, parameter_plot_end);
[plot_alpha_k_ssaga, plot_alpha_ssaga] = trim_trace(early_alpha_k_ssaga, mean_early_alpha_ssaga, parameter_plot_end);
[plot_alpha_k_ssarah, plot_alpha_ssarah] = trim_trace(early_alpha_k_ssarah, mean_early_alpha_ssarah, parameter_plot_end);
parameter_marker_step = max(1, ceil(parameter_plot_end / 40));

% 最终汇总使用原问题的训练目标 f(x)+mu*||y||_1，而非测试集纯损失
final_obj_bsarah = cellfun(@(v) v(end), all_obj_bsarah);
final_obj_bsvrg  = cellfun(@(v) v(end), all_obj_bsvrg);
final_obj_sadmm  = cellfun(@(v) v(end), all_obj_sadmm);
final_obj_ssvrg  = cellfun(@(v) v(end), all_obj_ssvrg);
final_obj_ssaga  = cellfun(@(v) v(end), all_obj_ssaga);
final_obj_ssarah = cellfun(@(v) v(end), all_obj_ssarah);


%% 5. 绘制横向五子图（训练目标、约束残差、测试损失、罚参数和步长）
summary_fig = figure('Name', sprintf('%s - Run %d: convergence summary', dataset_name, plot_run), ...
    'NumberTitle', 'off', 'Position', [20, 120, 1850, 620]);
idx_marker = round(linspace(1, grid_pts, 8)); 
% Insets magnify the clustered stochastic-method region without altering
% the main axes.  Their limits are derived from the active data set.
inset_time_range = [min(2, max_time), min(12, max_time)];
if inset_time_range(1) >= inset_time_range(2)
    inset_time_range(1) = 0;
end
inset_mask = mean_t >= inset_time_range(1) & mean_t <= inset_time_range(2);
inset_train_values = [mean_obj_bsarah(inset_mask); mean_obj_bsvrg(inset_mask); ...
    mean_obj_ssvrg(inset_mask); mean_obj_ssaga(inset_mask); mean_obj_ssarah(inset_mask)];
inset_test_values = [mean_loss_bsarah(inset_mask); mean_loss_bsvrg(inset_mask); ...
    mean_loss_ssvrg(inset_mask); mean_loss_ssaga(inset_mask); mean_loss_ssarah(inset_mask)];
inset_train_range = padded_limits(inset_train_values);
inset_test_range = padded_limits(inset_test_values);

% ---------------- Left Subplot: Training Objective ----------------
train_ax = subplot(1, 5, 1);
% 采样步长：每隔若干数据点绘制一个标记，避免标记过密
sample_step = 10;  

% 绘图：只在间隔采样点显示marker，其余只画线
p1 = plot(mean_t, mean_obj_bsarah, 'g--', ...
    'Marker','d','MarkerIndices',1:sample_step:length(mean_t),'LineWidth',1); hold on;
p2 = plot(mean_t, mean_obj_bsvrg, '--', 'Color', [0.85, 0.33, 0.10], ...
    'Marker','p','MarkerIndices',1:sample_step:length(mean_t),'LineWidth',1);
p3 = plot(mean_t, mean_obj_sadmm, 'c--', ...
    'Marker','v','MarkerIndices',1:sample_step:length(mean_t),'LineWidth',1);
p4 = plot(mean_t, mean_obj_ssvrg, 'b-', ...
    'Marker','s','MarkerIndices',1:sample_step:length(mean_t),'LineWidth',1);
p5 = plot(mean_t, mean_obj_ssaga, 'm-', ...
    'Marker','o','MarkerIndices',1:sample_step:length(mean_t),'LineWidth',1);
p6 = plot(mean_t, mean_obj_ssarah, 'r-', ...
    'Marker','^','MarkerIndices',1:sample_step:length(mean_t),'LineWidth',1);

% plot(mean_t(idx_marker), mean_obj_bsarah(idx_marker), 'go', 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'LineWidth', 1.5);
% plot(mean_t(idx_marker), mean_obj_ssvrg(idx_marker), 'bs', 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'LineWidth', 1.5);
% plot(mean_t(idx_marker), mean_obj_ssaga(idx_marker), 'md', 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'LineWidth', 1.5);
% plot(mean_t(idx_marker), mean_obj_ssarah(idx_marker), 'r*', 'MarkerSize', 7, 'LineWidth', 1.5); 
xticks(0:10:max_time);
xlabel('time (sec.)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('objective function value', 'FontSize', 12, 'FontWeight', 'bold');
title('Training Set Convergence', 'FontSize', 13);
xlim([0, max_time]); grid on;
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.6, 'FontSize', 11);

% ---------------- Right Subplot: Test Loss ----------------
test_ax = subplot(1, 5, 3);
sample_step = 10;
% p2 = plot(mean_t, mean_loss_bsarah, 'g-', ...
%     'Marker','d','MarkerIndices',1:sample_step:length(mean_t),'LineWidth',1);
% p3 = plot(mean_t, mean_loss_ssvrg, 'b-', ...
%     'Marker','s','MarkerIndices',1:sample_step:length(mean_t),'LineWidth',1);
% p4 = plot(mean_t, mean_loss_ssaga, 'm-', ...
%     'Marker','o','MarkerIndices',1:sample_step:length(mean_t),'LineWidth',1);
% p5 = plot(mean_t, mean_loss_ssarah, 'r-', ...
%     'Marker','^','MarkerIndices',1:sample_step:length(mean_t),'LineWidth',1);
p1 = semilogy(mean_t, max(mean_loss_bsarah, realmin), 'g--', ...
    'Marker','d','MarkerIndices',1:sample_step:length(mean_t),'LineWidth',1); hold on;
p2 = semilogy(mean_t, max(mean_loss_bsvrg, realmin), '--', 'Color', [0.85, 0.33, 0.10], ...
    'Marker','p','MarkerIndices',1:sample_step:length(mean_t),'LineWidth',1);
p3 = semilogy(mean_t, max(mean_loss_sadmm, realmin), 'c--', ...
    'Marker','v','MarkerIndices',1:sample_step:length(mean_t),'LineWidth',1);
p4 = semilogy(mean_t, max(mean_loss_ssvrg, realmin), 'b-', ...
    'Marker','s','MarkerIndices',1:sample_step:length(mean_t),'LineWidth',1);
p5 = semilogy(mean_t, max(mean_loss_ssaga, realmin), 'm-', ...
    'Marker','o','MarkerIndices',1:sample_step:length(mean_t),'LineWidth',1);
p6 = semilogy(mean_t, max(mean_loss_ssarah, realmin), 'r-', ...
    'Marker','^','MarkerIndices',1:sample_step:length(mean_t),'LineWidth',1);


% plot(mean_t(idx_marker), mean_loss_bsarah(idx_marker), 'go', 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'LineWidth', 1.5);
% plot(mean_t(idx_marker), mean_loss_ssvrg(idx_marker), 'bs', 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'LineWidth', 1.5);
% plot(mean_t(idx_marker), mean_loss_ssaga(idx_marker), 'md', 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'LineWidth', 1.5);
% plot(mean_t(idx_marker), mean_loss_ssarah(idx_marker), 'r*', 'MarkerSize', 7, 'LineWidth', 1.5); 

xticks(0:10:max_time);
xlabel('time (sec.)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('test loss (log scale)', 'FontSize', 12, 'FontWeight', 'bold');
title('Test Set Generalization', 'FontSize', 13);
xlim([0, max_time]); grid on;
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.6, 'FontSize', 11);

%% 6. 绘制自适应罚参数和步长的变化轨迹
figure(summary_fig);
beta_ax = subplot(1, 5, 4);
p_beta_svrg = plot(plot_k_ssvrg, plot_beta_ssvrg, 'b-s', ...
    'LineWidth', 1.4, 'MarkerSize', 5, 'MarkerIndices', 1:parameter_marker_step:length(plot_k_ssvrg), ...
    'MarkerFaceColor', 'w'); hold on;
p_beta_saga = plot(plot_k_ssaga, plot_beta_ssaga, 'm-o', ...
    'LineWidth', 1.4, 'MarkerSize', 5, 'MarkerIndices', 1:parameter_marker_step:length(plot_k_ssaga), ...
    'MarkerFaceColor', 'w');
p_beta_sarah = plot(plot_k_ssarah, plot_beta_ssarah, 'r-.^', ...
    'LineWidth', 1.4, 'MarkerSize', 5, 'MarkerIndices', 1:parameter_marker_step:length(plot_k_ssarah), ...
    'MarkerFaceColor', 'w');
xlabel('inner iteration k', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('\beta_k', 'FontSize', 12, 'FontWeight', 'bold');
title('Penalty Adjustment Until 50-Step Beta Stabilization', 'FontSize', 13);
xlim([0, parameter_plot_end]);
ylim(padded_limits([plot_beta_ssvrg; plot_beta_ssaga; plot_beta_ssarah]));
grid on;
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.6, 'FontSize', 11);

alpha_ax = subplot(1, 5, 5);
p_alpha_svrg = plot(plot_alpha_k_ssvrg, plot_alpha_ssvrg, 'b-s', ...
    'LineWidth', 1.4, 'MarkerSize', 5, 'MarkerIndices', 1:parameter_marker_step:length(plot_alpha_k_ssvrg), ...
    'MarkerFaceColor', 'w'); hold on;
p_alpha_saga = plot(plot_alpha_k_ssaga, plot_alpha_ssaga, 'm-o', ...
    'LineWidth', 1.4, 'MarkerSize', 5, 'MarkerIndices', 1:parameter_marker_step:length(plot_alpha_k_ssaga), ...
    'MarkerFaceColor', 'w');
p_alpha_sarah = plot(plot_alpha_k_ssarah, plot_alpha_ssarah, 'r-.^', ...
    'LineWidth', 1.4, 'MarkerSize', 5, 'MarkerIndices', 1:parameter_marker_step:length(plot_alpha_k_ssarah), ...
    'MarkerFaceColor', 'w');
xlabel('inner iteration k', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('\alpha_k', 'FontSize', 12, 'FontWeight', 'bold');
title('Step Size Shown Until 50-Step Beta Stabilization', 'FontSize', 13);
xlim([0, parameter_plot_end]);
ylim(padded_limits([plot_alpha_ssvrg; plot_alpha_ssaga; plot_alpha_ssarah]));
grid on;
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.6, 'FontSize', 11);

%% 7. 中间子图：Constraint residual versus elapsed time
% 复用第 5 节创建的汇总窗口，将约束残差放在训练和测试曲线之间
figure(summary_fig);
residual_ax = subplot(1, 5, 2);
residual_marker_step = 10;
if Num_Runs >= 1
    % Plot raw recorded points for one experiment, so the reported minimum
    % residual and its star marker use exactly the same coordinates.
    [plot_t_bsarah, plot_res_bsarah] = recorded_residual_trace(all_t_bsarah{plot_run}, all_r_bsarah{plot_run});
    [plot_t_bsvrg, plot_res_bsvrg] = recorded_residual_trace(all_t_bsvrg{plot_run}, all_r_bsvrg{plot_run});
    [plot_t_sadmm, plot_res_sadmm] = recorded_residual_trace(all_t_sadmm{plot_run}, all_r_sadmm{plot_run});
    [plot_t_ssvrg, plot_res_ssvrg] = recorded_residual_trace(all_t_ssvrg{plot_run}, all_r_ssvrg{plot_run});
    [plot_t_ssaga, plot_res_ssaga] = recorded_residual_trace(all_t_ssaga{plot_run}, all_r_ssaga{plot_run});
    [plot_t_ssarah, plot_res_ssarah] = recorded_residual_trace(all_t_ssarah{plot_run}, all_r_ssarah{plot_run});
    plot_best_time_bsarah = best_time_bsarah(plot_run); plot_best_res_bsarah = best_res_bsarah(plot_run);
    plot_best_time_bsvrg = best_time_bsvrg(plot_run); plot_best_res_bsvrg = best_res_bsvrg(plot_run);
    plot_best_time_sadmm = best_time_sadmm(plot_run); plot_best_res_sadmm = best_res_sadmm(plot_run);
    plot_best_time_ssvrg = best_time_ssvrg(plot_run); plot_best_res_ssvrg = best_res_ssvrg(plot_run);
    plot_best_time_ssaga = best_time_ssaga(plot_run); plot_best_res_ssaga = best_res_ssaga(plot_run);
    plot_best_time_ssarah = best_time_ssarah(plot_run); plot_best_res_ssarah = best_res_ssarah(plot_run);
else
    plot_t_bsarah = residual_time; plot_res_bsarah = mean_res_bsarah;
    plot_t_bsvrg = residual_time;  plot_res_bsvrg = mean_res_bsvrg;
    plot_t_sadmm = residual_time;  plot_res_sadmm = mean_res_sadmm;
    plot_t_ssvrg = residual_time;  plot_res_ssvrg = mean_res_ssvrg;
    plot_t_ssaga = residual_time;  plot_res_ssaga = mean_res_ssaga;
    plot_t_ssarah = residual_time; plot_res_ssarah = mean_res_ssarah;
    plot_best_time_bsarah = mean(best_time_bsarah); plot_best_res_bsarah = mean(best_res_bsarah);
    plot_best_time_bsvrg = mean(best_time_bsvrg); plot_best_res_bsvrg = mean(best_res_bsvrg);
    plot_best_time_sadmm = mean(best_time_sadmm); plot_best_res_sadmm = mean(best_res_sadmm);
    plot_best_time_ssvrg = mean(best_time_ssvrg); plot_best_res_ssvrg = mean(best_res_ssvrg);
    plot_best_time_ssaga = mean(best_time_ssaga); plot_best_res_ssaga = mean(best_res_ssaga);
    plot_best_time_ssarah = mean(best_time_ssarah); plot_best_res_ssarah = mean(best_res_ssarah);
end
p_res_bsarah = semilogy(plot_t_bsarah, max(plot_res_bsarah, realmin), 'g--', ...
    'Marker', 'd', 'MarkerIndices', 1:residual_marker_step:length(plot_t_bsarah), 'LineWidth', 1.2); hold on;
p_res_bsvrg = semilogy(plot_t_bsvrg, max(plot_res_bsvrg, realmin), '--', 'Color', [0.85, 0.33, 0.10], ...
    'Marker', 'p', 'MarkerIndices', 1:residual_marker_step:length(plot_t_bsvrg), 'LineWidth', 1.2);
p_res_sadmm = semilogy(plot_t_sadmm, max(plot_res_sadmm, realmin), 'c--', ...
    'Marker', 'v', 'MarkerIndices', 1:residual_marker_step:length(plot_t_sadmm), 'LineWidth', 1.2);
p_res_ssvrg = semilogy(plot_t_ssvrg, max(plot_res_ssvrg, realmin), 'b-', ...
    'Marker', 's', 'MarkerIndices', 1:residual_marker_step:length(plot_t_ssvrg), 'LineWidth', 1.2);
p_res_ssaga = semilogy(plot_t_ssaga, max(plot_res_ssaga, realmin), 'm-', ...
    'Marker', 'o', 'MarkerIndices', 1:residual_marker_step:length(plot_t_ssaga), 'LineWidth', 1.2);
p_res_ssarah = semilogy(plot_t_ssarah, max(plot_res_ssarah, realmin), 'r-', ...
    'Marker', '^', 'MarkerIndices', 1:residual_marker_step:length(plot_t_ssarah), 'LineWidth', 1.2);

% Yellow hexagons identify the minimum recorded residual of each method.
p_min_bsarah = semilogy(plot_best_time_bsarah, plot_best_res_bsarah, 'h', ...
    'Color', 'g', 'MarkerFaceColor', 'y', 'MarkerSize', 10, 'LineWidth', 1.2);
p_min_bsvrg = semilogy(plot_best_time_bsvrg, plot_best_res_bsvrg, 'h', ...
    'Color', [0.85, 0.33, 0.10], 'MarkerFaceColor', 'y', 'MarkerSize', 10, 'LineWidth', 1.2);
p_min_sadmm = semilogy(plot_best_time_sadmm, plot_best_res_sadmm, 'h', ...
    'Color', 'c', 'MarkerFaceColor', 'y', 'MarkerSize', 10, 'LineWidth', 1.2);
p_min_ssvrg = semilogy(plot_best_time_ssvrg, plot_best_res_ssvrg, 'h', ...
    'Color', 'b', 'MarkerFaceColor', 'y', 'MarkerSize', 10, 'LineWidth', 1.2);
p_min_ssaga = semilogy(plot_best_time_ssaga, plot_best_res_ssaga, 'h', ...
    'Color', 'm', 'MarkerFaceColor', 'y', 'MarkerSize', 10, 'LineWidth', 1.2);
p_min_ssarah = semilogy(plot_best_time_ssarah, plot_best_res_ssarah, 'h', ...
    'Color', 'r', 'MarkerFaceColor', 'y', 'MarkerSize', 10, 'LineWidth', 1.2);
summary_legend = legend([p_res_sadmm, p_res_bsvrg, p_res_bsarah, ...
    p_res_ssvrg, p_res_ssaga, p_res_ssarah, p_min_bsarah], ...
    'SADMM', 'SVRG-ADMM', 'SARAH-ADMM', 'Alg.1-svrg', ...
    'Alg.1-saga', 'Alg.1-sarah', 'minimum residual', ...
    'Orientation', 'horizontal', 'NumColumns', 7);
% 将统一图例放在五张子图下方，并相对于整个汇总窗口居中
set(summary_legend, 'Units', 'normalized', ...
    'Position', [0.20, 0.025, 0.60, 0.045], 'FontSize', 9, ...
    'Box', 'off', 'Color', 'none');
xticks(0:10:max_time);
xlabel('time (sec.)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('primal residual ||Ax-y||_1 (log scale)', 'FontSize', 12, 'FontWeight', 'bold');
title('Constraint Residual Convergence', 'FontSize', 13);
xlim([residual_time(1), max_time]); grid on;
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.6, 'FontSize', 11);


% Reserve a dedicated bottom band for the shared legend and x-axis labels.
main_axes = [train_ax, residual_ax, test_ax, beta_ax, alpha_ax];
for axes_index = 1:numel(main_axes)
    axes_position = get(main_axes(axes_index), 'Position');
    axes_position(2) = 0.24;
    axes_position(4) = 0.62;
    set(main_axes(axes_index), 'Position', axes_position);
end

% Restore the local zoom panels inside the first and third subplots.
train_position = get(train_ax, 'Position');
train_inset_position = [train_position(1) + 0.47 * train_position(3), ...
    train_position(2) + 0.08 * train_position(4), ...
    0.49 * train_position(3), 0.37 * train_position(4)];
train_inset_ax = axes('Parent', summary_fig, 'Position', train_inset_position, ...
    'Box', 'on', 'FontSize', 7, 'HandleVisibility', 'off');
plot(train_inset_ax, mean_t, mean_obj_bsarah, 'g--', 'LineWidth', 0.9); hold(train_inset_ax, 'on');
plot(train_inset_ax, mean_t, mean_obj_bsvrg, '--', 'Color', [0.85, 0.33, 0.10], 'LineWidth', 0.9);
plot(train_inset_ax, mean_t, mean_obj_ssvrg, 'b-', 'LineWidth', 0.9);
plot(train_inset_ax, mean_t, mean_obj_ssaga, 'm-', 'LineWidth', 0.9);
plot(train_inset_ax, mean_t, mean_obj_ssarah, 'r-', 'LineWidth', 0.9);
xlim(train_inset_ax, inset_time_range);
ylim(train_inset_ax, inset_train_range);
title(train_inset_ax, 'Zoom', 'FontSize', 7, 'FontWeight', 'normal');
grid(train_inset_ax, 'on');
set(train_inset_ax, 'GridLineStyle', ':', 'GridAlpha', 0.5);

test_position = get(test_ax, 'Position');
test_inset_position = [test_position(1) + 0.47 * test_position(3), ...
    test_position(2) + 0.08 * test_position(4), ...
    0.49 * test_position(3), 0.37 * test_position(4)];
test_inset_ax = axes('Parent', summary_fig, 'Position', test_inset_position, ...
    'Box', 'on', 'FontSize', 7, 'HandleVisibility', 'off');
semilogy(test_inset_ax, mean_t, max(mean_loss_bsarah, realmin), 'g--', 'LineWidth', 0.9); hold(test_inset_ax, 'on');
semilogy(test_inset_ax, mean_t, max(mean_loss_bsvrg, realmin), '--', 'Color', [0.85, 0.33, 0.10], 'LineWidth', 0.9);
semilogy(test_inset_ax, mean_t, max(mean_loss_ssvrg, realmin), 'b-', 'LineWidth', 0.9);
semilogy(test_inset_ax, mean_t, max(mean_loss_ssaga, realmin), 'm-', 'LineWidth', 0.9);
semilogy(test_inset_ax, mean_t, max(mean_loss_ssarah, realmin), 'r-', 'LineWidth', 0.9);
xlim(test_inset_ax, inset_time_range);
ylim(test_inset_ax, inset_test_range);
title(test_inset_ax, 'Zoom', 'FontSize', 7, 'FontWeight', 'normal');
grid(test_inset_ax, 'on');
set(test_inset_ax, 'GridLineStyle', ':', 'GridAlpha', 0.5);

end

%% 8. 打印指标汇总
fprintf('\n========================================================================\n');
fprintf('        各算法限时运行 %d 秒，共 %d 次独立实验（逐次输出，不取平均）\n', max_time, Num_Runs);
fprintf('========================================================================\n');
algorithm_names = {'SADMM', 'SVRG-ADMM', 'SARAH-ADMM', ...
                   'Alg.1-svrg', 'Alg.1-saga', 'Alg.1-sarah'};
objective_values = [final_obj_sadmm, final_obj_bsvrg, final_obj_bsarah, ...
                    final_obj_ssvrg, final_obj_ssaga, final_obj_ssarah];
final_residual_values = [res_final_sadmm, res_final_bsvrg, res_final_bsarah, ...
                         res_final_ssvrg, res_final_ssaga, res_final_ssarah];
minimum_residual_values = [best_res_sadmm, best_res_bsvrg, best_res_bsarah, ...
                           best_res_ssvrg, best_res_ssaga, best_res_ssarah];
objective_at_min_values = [best_obj_sadmm, best_obj_bsvrg, best_obj_bsarah, ...
                           best_obj_ssvrg, best_obj_ssaga, best_obj_ssarah];
time_at_min_values = [best_time_sadmm, best_time_bsvrg, best_time_bsarah, ...
                      best_time_ssvrg, best_time_ssaga, best_time_ssarah];
for run = 1:Num_Runs
    fprintf('\n%% Run %d / %d\n', run, Num_Runs);
    for algorithm_id = 1:numel(algorithm_names)
        fprintf('&%s &%.6f &%.3e &%.3e &%.6f &%.3f\\\\\n', ...
            algorithm_names{algorithm_id}, objective_values(run, algorithm_id), ...
            final_residual_values(run, algorithm_id), minimum_residual_values(run, algorithm_id), ...
            objective_at_min_values(run, algorithm_id), time_at_min_values(run, algorithm_id));
    end
end
fprintf('========================================================================\n');

algorithm_order = [3, 2, 1, 4, 5, 6];
fprintf('\n分类指标（测试集）\n');
fprintf('%-20s | %-18s | %-18s\n', '算法名称', '分类错误率', 'Balanced accuracy');
fprintf('------------------------------------------------------------------------\n');
for run = 1:Num_Runs
    fprintf('Run %d / %d\n', run, Num_Runs);
    for display_id = 1:numel(algorithm_names)
        algorithm_id = algorithm_order(display_id);
        fprintf('%-20s | %18.6f | %18.6f\n', ...
            algorithm_names{display_id}, ...
            class_error(run, algorithm_id), balanced_accuracy(run, algorithm_id));
    end
end
fprintf('========================================================================\n');

fprintf('完成数据集：%s\n', dataset_file);
fprintf('############################################################\n');
end  % dataset_index

%% ========================================================================
%  辅助插值与对齐函数
% ========================================================================
function mean_f = interp_and_average(all_t, all_f, target_t)
    num_runs = length(all_t);
    grid_pts = length(target_t);
    interp_matrix = zeros(grid_pts, num_runs);
    for r = 1:num_runs
        t_orig = all_t{r};
        f_orig = all_f{r};
        [t_orig, u_idx] = unique(t_orig);
        f_orig = f_orig(u_idx);
        valid = isfinite(t_orig) & isfinite(f_orig) & t_orig >= 0 & t_orig <= target_t(end);
        t_orig = t_orig(valid);
        f_orig = f_orig(valid);
        if isempty(t_orig)
            error('第 %d 次运行没有有效的时间—指标记录。', r);
        elseif numel(t_orig) == 1
            interp_matrix(:, r) = f_orig(1);
        else
            interp_matrix(:, r) = interp1(t_orig, f_orig, target_t, 'linear', f_orig(end));
        end
    end
    mean_f = mean(interp_matrix, 2);
end

function mean_f = interp_residual_and_average(all_t, all_residual, target_t)
    % The t=0 point is exactly feasible by initialization (x0=y0=0), not
    % an observed convergence result.  Do not extrapolate it over time.
    num_runs = length(all_t);
    interp_matrix = nan(length(target_t), num_runs);
    for r = 1:num_runs
        t_orig = all_t{r}(:);
        residual_orig = all_residual{r}(:);
        valid = isfinite(t_orig) & isfinite(residual_orig) & t_orig > 0;
        t_orig = t_orig(valid);
        residual_orig = residual_orig(valid);
        if isempty(t_orig)
            continue;
        end
        [t_orig, unique_idx] = unique(t_orig);
        residual_orig = residual_orig(unique_idx);
        if numel(t_orig) == 1
            interp_matrix(:, r) = residual_orig(1);
        else
            interp_matrix(:, r) = interp1(t_orig, residual_orig, ...
                target_t, 'linear', residual_orig(end));
        end
    end
    mean_f = mean(interp_matrix, 2, 'omitnan');
end

function [t, residual] = recorded_residual_trace(t_history, residual_history)
    t = t_history(:);
    residual = residual_history(:);
    valid = isfinite(t) & isfinite(residual) & t > 0;
    t = t(valid);
    residual = residual(valid);
end

function [iteration_grid, mean_value] = average_by_iteration(all_k, all_value)
    num_runs = length(all_k);
    has_valid_trace = false(num_runs, 1);
    final_iterations = zeros(num_runs, 1);
    for r = 1:num_runs
        k = all_k{r}(:);
        value = all_value{r}(:);
        valid = isfinite(k) & isfinite(value) & k >= 0;
        if any(valid)
            has_valid_trace(r) = true;
            final_iterations(r) = max(k(valid));
        end
    end

    % A costly full-gradient step can exhaust the time budget before the
    % first parameter update on very large data sets.  Return an empty
    % trace in that case so the diagnostic plot does not stop the run.
    if ~any(has_valid_trace)
        iteration_grid = zeros(0, 1);
        mean_value = zeros(0, 1);
        return;
    end

    all_k = all_k(has_valid_trace);
    all_value = all_value(has_valid_trace);
    final_iterations = final_iterations(has_valid_trace);
    num_runs = numel(all_k);
    common_final_iteration = min(final_iterations);
    grid_pts = min(1000, common_final_iteration + 1);
    iteration_grid = unique(round(linspace(0, common_final_iteration, grid_pts)))';
    interp_matrix = zeros(length(iteration_grid), num_runs);

    for r = 1:num_runs
        k_orig = all_k{r};
        value_orig = all_value{r};
        [k_orig, unique_idx] = unique(k_orig);
        value_orig = value_orig(unique_idx);
        valid = isfinite(k_orig) & isfinite(value_orig) & ...
            k_orig >= 0 & k_orig <= common_final_iteration;
        k_orig = k_orig(valid);
        value_orig = value_orig(valid);
        if isempty(k_orig)
            error('第 %d 次运行没有有效的迭代—参数记录。', r);
        elseif numel(k_orig) == 1
            interp_matrix(:, r) = value_orig(1);
        else
            interp_matrix(:, r) = interp1(k_orig, value_orig, ...
                iteration_grid, 'linear');
        end
    end
    mean_value = mean(interp_matrix, 2);
end

function final_k = first_beta_window_stability_k(k, beta, tolerance, window_length)
    k = k(:);
    beta = beta(:);
    valid = isfinite(k) & isfinite(beta);
    k = k(valid);
    beta = beta(valid);
    if isempty(k)
        final_k = 0;
        return;
    end
    window_length = max(1, floor(window_length));
    if numel(k) < window_length
        final_k = k(end);
        return;
    end
    for window_end = window_length:numel(k)
        window_indices = window_end - window_length + 1:window_end;
        beta_deviation = max(abs(beta(window_indices) - beta(window_end)));
        if beta_deviation <= tolerance
            final_k = k(window_end);
            return;
        end
    end
    % No stable 50-step window was observed within the recorded trace.
    final_k = k(end);
end

function [k_trimmed, value_trimmed] = trim_trace(k, value, final_k)
    keep = k <= final_k;
    k_trimmed = k(keep);
    value_trimmed = value(keep);
end

function [value, index] = minimum_residual_and_index(residual_history)
    valid_indices = find(isfinite(residual_history));
    valid_indices = valid_indices(valid_indices > 1);
    if isempty(valid_indices)
        value = NaN;
        index = NaN;
        return;
    end
    [value, local_index] = min(residual_history(valid_indices));
    index = valid_indices(local_index);
end

function value = recorded_value_at_index(history, index)
    if ~isfinite(index) || index < 1 || index > numel(history)
        value = NaN;
    else
        value = history(index);
    end
end

function limits = padded_limits(values)
    values = values(isfinite(values));
    if isempty(values)
        limits = [0, 1];
        return;
    end
    value_min = min(values);
    value_max = max(values);
    value_span = value_max - value_min;
    margin = max(0.08 * value_span, ...
        max(1e-4, 0.005 * max(1, max(abs(values)))));
    limits = [value_min - margin, value_max + margin];
end




