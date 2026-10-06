function divp = divergence_periodic(px, py)
%DIVERGENCE_PERIODIC Adjoint of gradient_periodic under circular boundary.
divp = px - circshift(px, 1, 1) + py - circshift(py, 1, 2);
end
