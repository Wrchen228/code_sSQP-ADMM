function kernel = load_section52_texton(path, targetSize, opts)
%LOAD_SECTION52_TEXTON Load and normalize the paper's stationary-noise texton.
%
% The PDF-extracted kernels are stored as full images in data/paper_pdf.
% This helper crops any margin, resizes to the requested support, and then
% applies a chosen normalization.
if nargin < 3
    opts = struct();
end
if ~isfield(opts, 'normalize')
    opts.normalize = 'sum';
end
if ~isfield(opts, 'cropRect')
    opts.cropRect = [];
end
if ~isfield(opts, 'cropContent')
    opts.cropContent = true;
end
if ~isfield(opts, 'preserveAspect')
    opts.preserveAspect = false;
end

img = imread(path);
if ndims(img) == 3
    img = rgb2gray(img);
end
img = im2double(img);

if ~isempty(opts.cropRect)
    rect = round(opts.cropRect);
    x1 = max(1, rect(1));
    y1 = max(1, rect(2));
    x2 = min(size(img, 2), x1 + rect(3) - 1);
    y2 = min(size(img, 1), y1 + rect(4) - 1);
    img = img(y1:y2, x1:x2);
end

if opts.cropContent
    img = crop_to_content(img, 0.01);
end
kernel = img;

if nargin >= 2 && ~isempty(targetSize) && any(size(kernel) ~= targetSize)
    if opts.preserveAspect
        kernel = local_resize_with_aspect(kernel, targetSize);
    else
        kernel = imresize(kernel, targetSize, 'bicubic');
    end
end

switch lower(opts.normalize)
    case 'sum'
        s = sum(kernel(:));
        if abs(s) > eps
            kernel = kernel / s;
        end
    case 'fro'
        nrm = norm(kernel(:));
        if nrm > eps
            kernel = kernel / nrm;
        end
    case 'none'
        % Keep the resized texton as is.
    otherwise
        error('Unknown normalization mode: %s', opts.normalize);
end

function out = local_resize_with_aspect(img, targetSize)
targetH = targetSize(1);
targetW = targetSize(2);
srcH = size(img, 1);
srcW = size(img, 2);
scale = min(targetH / srcH, targetW / srcW);
newH = max(1, round(srcH * scale));
newW = max(1, round(srcW * scale));
resized = imresize(img, [newH, newW], 'bicubic');
out = zeros(targetH, targetW);
r0 = floor((targetH - newH) / 2) + 1;
c0 = floor((targetW - newW) / 2) + 1;
out(r0:r0 + newH - 1, c0:c0 + newW - 1) = resized;
end
end
