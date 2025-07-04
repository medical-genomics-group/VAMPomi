# gVAMPomi DNAnexus applet
## Build the App
First, the binary executables (`main` and `main_probit` for the linear and probit model, respectively) compiled on the DNAnexus machine must be placed in the following directory:
``
gvampomi/resources/usr/bin
``
Second, the gVAMPomi applet can be built using the dx API as follows:
```
dx build gvampomi
```

## Example
The following input options must be specified at a minimum to run in inference mode:

```
PID=... #DNAnexus project ID

OUTDIR=${PID}:/...

dx run gvampomi \
  -idata_file=${PID}:/... \
  -iprobe_file=${PID}:/... \
  -iphen_file=${PID}:/... \
  -iout=example \
  -irun_mode=infere \
  --destination=${OUTDIR} \
```

DNAnexus documentation:
https://documentation.dnanexus.com/developer/apps/intro-to-building-apps

Input files:
- ``data_file`` Path to .bin file including the data in binary format.
- ``probe_file`` List of names assigned to markers (e.g. probe names, genes,...), each name in separate line. The length has to corespond to number of columns in ``data_file``.
- ``phen_file`` Path to file containing phenotype data in PLINK format.
- ``cov_file`` Path to file containing covariate data in PLINK format.
- ``model`` linear/bin_class.
- ``iterations`` Number of VAMP iterations.
- ``rho`` Damping factor (learning rate).
- ``prior_probs`` Initial prior mixture coefficients (separated by comma, must sum up to 1).
- ``prior_vars`` Initial prior variances (separated by comma).
- ``stop_criteria_thr`` Criterion for early stopping.
- ``out`` Output files prefix.

Output files:
- ``example.gvamp`` Data frame including estimated marker effects, Posterior Inclusion Probabilities (PIP), and P-values.
- ``example.log`` gVAMPomi log file.
- ``example_summary.csv`` Data frame containing estimated marker variance and number of discoveries.
