function mask = sample_batch_mask(sz, batchSize)
%SAMPLE_BATCH_MASK Random logical mask with a fixed number of active pixels.
n = prod(sz);
batchSize = min(max(1, round(batchSize)), n);
idx = randperm(n, batchSize);
mask = false(sz);
mask(idx) = true;
end
