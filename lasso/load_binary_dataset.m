function [X, y] = load_binary_dataset(dataset_file)
%LOAD_BINARY_DATASET 加载并校验二分类数据集。
% 支持 X/y 和历史文件中的 X_toal/Y_toal 两种变量命名。

try
    data = load(dataset_file);
    if isfield(data, 'X') && isfield(data, 'y')
        X = data.X;
        y = data.y;
    elseif isfield(data, 'X_toal') && isfield(data, 'Y_toal')
        X = data.X_toal;
        y = data.Y_toal;
    else
        error('%s 中未找到 X/y 或 X_toal/Y_toal。', dataset_file);
    end
catch load_error
    % HIGGS is stored as a generic HDF5 container to permit streamed
    % conversion of its 11-million-row data set.
    try
        X = h5read(dataset_file, '/X');
        y = h5read(dataset_file, '/y');
    catch
        rethrow(load_error);
    end
end

y = y(:);
if size(X, 1) ~= numel(y)
    error('%s 的特征行数与标签数不一致。', dataset_file);
end

labels = unique(y);
if numel(labels) ~= 2
    error('%s 不是二分类数据集，共发现 %d 个标签。', dataset_file, numel(labels));
end
if ~isequal(labels, [-1; 1])
    original_y = y;
    y = -ones(size(original_y));
    y(original_y == labels(2)) = 1;
    fprintf('已将标签 [%g, %g] 映射为 [-1, 1]。\n', labels(1), labels(2));
end

end
