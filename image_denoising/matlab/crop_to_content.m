function y = crop_to_content(x, tolerance)
%CROP_TO_CONTENT Crop uniform border around a figure-extracted image.
if nargin < 2
    tolerance = 0.01;
end

if ndims(x) == 3
    x = rgb2gray(x);
end
x = im2double(x);

border = [x(1, :), x(end, :), x(:, 1).', x(:, end).'];
bg = median(border(:));
mask = abs(x - bg) > tolerance;

if ~any(mask(:))
    y = x;
    return;
end

[r, c] = find(mask);
pad = 2;
r1 = max(min(r) - pad, 1);
r2 = min(max(r) + pad, size(x, 1));
c1 = max(min(c) - pad, 1);
c2 = min(max(c) + pad, size(x, 2));
y = x(r1:r2, c1:c2);
end
