function [dx, dy] = gradient_periodic(x)
%GRADIENT_PERIODIC Forward differences with circular boundary conditions.
dx = circshift(x, -1, 1) - x;
dy = circshift(x, -1, 2) - x;
end
