# VAMPomi
VAMPomi is Vector Approximate Message Passing method for analyzing omics data, written in C++ (source code in src/).

This repository also includes scripts for data simulations and post-processing on methylation data.

simulation/sim_top_iid.py - loading adjusted and standardized methylation data from zarr files and simulate iid phenotype on the top.

# Example

```
module load gcc openmpi boost

vloc={Path to VAMPomi/src folder}

export OMP_NUM_THREADS={Number of OpenMP threads}

mpic++ ${vloc}/main_meth.cpp ${vloc}/vamp.cpp ${vloc}/utilities.cpp ${vloc}/data.cpp ${vloc}/options.cpp -march=native -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  ${vloc}/main_meth.exe

mpirun -np {Number of MPI ranks} ${vloc}/main_meth.exe [Input options]
```

# Input options

| Option | Description |
| --- | --- |
| `--run-mode` | `infere` / `test` |
| `--model` | Regression model `linear` / `bin_class` |
| `--meth-file` | Path to .bin file including the data in binary format |
| `--meth-file-test` | Path to .bin file for testing |
| `--phen-file` | Path to file containing phenotype data |
| `--phen-file-test` | Path to phenotype file for testing |
| `--cov-file` | Path to .cov file including covariates |
| `--cov-file-test` | Path to .cov file including covariates for testing |
| `--estimate-file` | Path to file including signal estimates for testing |
| `--N` | Number of individuals for inference |
| `--N-test` | Number of individuals for testing |
| `--Mt-test` | Total number of markers for testing |
| `--Mt` | Total number of markers for inference |
| `--C` | Indicates the number of covariates |
| `--out-dir` | Output directory for the signal estimates |
| `--out-name` | Name of the output file |
| `--iterations` | Maximal number of iterations |
| `--test-iter-range` | Iteration range for testing |
| `--rho` | Damping factor |
| `--use_lmmse_damp` | Indicates whether or not damping should be used in LMMSE step |
| `--gsm1` | Signal noise precision initialization |
| `--h2` | Heritability value used in simulations |
| `--CV` | Number of causal markers used in simulations |
| `--true_signal_file` | Path to the file containing true signals |
| `--num-mix-comp` | Nnumber of gaussian mixture components |
| `--probs` | Initial prior mixture coefficients (separated by comma, must sum up to 1) |
| `--vars` | Initial prior variances (separated by comma) |
| `--learn-vars` | Indicates whether or not prior variances are learned|
| `--CG-max-iter` | Maximal number of iteration used in conjugate gradient method |
| `--EM-err-thr` | Relative error threshold within expectation maximization |
| `--EM-max-iter` | Maximal number of iterations of expectation maximization |
| `--stop-criteria-thr` | Relative error threshold within expectation maximization |
| `--prior-tune-max-iter` | Maximal number of iterations of prior tuning |
| `--verbosity` | Whether or not to print out details |