# SQP-ADMM numerical experiments

This directory contains the MATLAB code used for the numerical experiments in the accompanying SQP-ADMM paper. Only the experiments reported in the paper are included.

## Requirements

- MATLAB (R2021b or later recommended)
- Image Processing Toolbox for the image-denoising experiments
- Git LFS when cloning or publishing the included MAT datasets

Run each experiment from its own directory so that relative data paths resolve correctly.

## 1. Graph-guided fused LASSO

The `lasso` directory reproduces the experiments on Adult, IJCNN, Covtype, and SUSY. It compares SADMM, SVRG-ADMM, SARAH-ADMM, and the SVRG/SAGA/SARAH variants of the proposed SQP-ADMM method.

```matlab
cd lasso
maint
```

The script uses the paper's 50/50 train-test split, deterministic random seeds, dataset-specific mini-batch sizes, and five independent runs. The full run can take a substantial amount of time, especially for SUSY. To run a subset, edit only the `dataset_files` list at the top of `maint.m`.

The SUSY MAT file is approximately 375 MB and is tracked through Git LFS because it exceeds the ordinary 100 MB per-file limit of common Git hosting services. Run `git lfs install` before the first commit/push, and clone with Git LFS enabled.

## 2. Image denoising

The `image_denoising` directory contains the Boat, Peppers, and Male cases reported in the paper. From MATLAB, run:

```matlab
cd image_denoising/matlab
run_section52_all_boat_peppers_male
```

Individual cases can be run with:

```matlab
run_section52_all_boat_50s
run_section52_all_peppers_200s
run_section52_all_male_500s
```

Generated figures and MAT files are written under `image_denoising/results/` and are ignored by Git.

## Repository contents

- `lasso/`: algorithms, dataset-specific parameters, four benchmark datasets, and the main reproduction script.
- `image_denoising/matlab/`: the three paper experiment entry points and their solver utilities.
- `image_denoising/data/`: the three source images and the two texture elements needed to generate stationary noise.

Exploratory parameter sweeps, diagnostic scripts, temporary files, logs, intermediate results, unrelated examples, and manuscript files are intentionally excluded.

## Reproducibility notes

The experiments are time-budget based, so small numerical differences can occur across MATLAB releases and hardware. Random seeds are fixed in the scripts. No manuscript source files are required to run the code.
