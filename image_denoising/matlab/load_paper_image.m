function x = load_paper_image(path, targetSize, doCrop)
%LOAD_PAPER_IMAGE Read a grayscale image and optionally crop/resize it.
if nargin < 3
    doCrop = true;
end
if nargin < 2
    targetSize = [];
end

img = imread(path);
if ndims(img) == 3
    img = rgb2gray(img);
end
img = im2double(img);
if doCrop
    img = crop_to_content(img, 0.01);
end
if ~isempty(targetSize) && any(size(img) ~= targetSize)
    img = imresize(img, targetSize, 'bicubic');
end
x = min(max(img, 0), 1);
end
