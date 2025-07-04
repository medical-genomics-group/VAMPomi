import numpy as np
import argparse
import os
import struct
import random
import pandas as pd
from scipy.stats import norm

print("---- Simulating example probit outcome with covariates ----\n", flush=True)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-out_dir", "--out-dir", help = "Output directory")
parser.add_argument("-out_name", "--out-name", help="Output file name", default="example")
parser.add_argument("-N", "--N", help="Number of individuals", default=2000)
parser.add_argument("-M", "--M", help="Number of markers", default=2000)
parser.add_argument("-C", "--C", help="Number of covariates", default=10)
parser.add_argument("-lam", "--lam", help="Sparsity", default=0.1)
parser.add_argument("-probit_thr", "--probit-thr", help="Probit threshold", default=0.5)
parser.add_argument("-epi_var", "--epi-var", help="Epigenetic component variance", default=0.5)
parser.add_argument("-cov_var", "--cov-var", help="Covariate component variance", default=0.3)
args = parser.parse_args()

out_name = args.out_name
out_dir = args.out_dir
N = int(args.N)
M = int(args.M)
C = int(args.C)
lam = float(args.lam)
probit_thr = float(args.probit_thr)
epi_var = float(args.epi_var)
cov_var = float(args.cov_var)

print("Input arguments:")
print("--out_dir", out_dir)
print("--out_name", out_name)
print("--N", N)
print("--M", M)
print("--C", C)
print("--lam", lam)
print("--probit-thr", probit_thr)
print("--cov-var", cov_var)
print("--epi-var", epi_var, flush=True)

sample_ind = np.arange(0,N)

print("\n...Simulating design matrix", flush=True)
X = np.random.normal(0,1,N*M).reshape((N,M))

# Saving design matrix in binary format
print("\n...Saving design matrix to bin file")
bin_fpath = os.path.join(out_dir, "%s.bin" % out_name)
print(bin_fpath, flush=True)
binf = open(bin_fpath, "wb")
b = struct.pack(str(N*M)+'d', *X.transpose().ravel().squeeze())
binf.write(b)
binf.close()

probe_names = ["probe"+str(i) for i in range(1,M+1)]

print("\n...Simulating covariate matrix", flush=True)
Z = np.random.normal(0,1,N*C).reshape((N,C)) #covariate matrix
print(np.shape(Z))

# Saving covariates
print("\n...Saving covariate matrix to .cov file")
cov_fpath = os.path.join(out_dir, "%s.cov" % out_name)
print(cov_fpath, flush=True)

df_cov = pd.DataFrame({"IID": sample_ind,
                      "FID": sample_ind})
cov_names = ["cov"+str(i) for i in range(1,C+1)]
for i,name in enumerate(cov_names):
    df_cov[name] = Z[:,i]
df_cov.to_csv(cov_fpath, index=False, sep="\t")

print("\n...Simulating marker effects", flush=True)
CM = int(M * lam) # number of cuasal markers
sigma2 = epi_var / CM
idx = random.sample(range(M), CM)
beta = np.zeros(M)
beta[idx] = np.random.normal(0.0, np.sqrt(sigma2), CM)
print("sigma2 = ", sigma2, flush=True)

print("\n...Simulating covariate effects", flush=True)
sigma2_cov = cov_var / C
delta = np.random.normal(0.0, np.sqrt(sigma2_cov), C)

print("\n...Computing outcome variable", flush=True)
g = np.matmul(X,beta)
c = np.matmul(Z,delta)
latent = g + c + np.random.normal(0, np.sqrt( 1 - (np.var(g) + np.var(c))),  N) # adding Gaussian noise
p = norm.cdf(latent)
y = np.zeros(N)
y[p >= probit_thr] = 1

print("Var(latent) = ", np.var(latent))
print("Var(g) = ", np.var(g))
print("Var(c) = ", np.var(c))

# Saving phenotype data
print("\n...Saving outcome to .phen file")
phen_fpath = os.path.join(out_dir, "%s.phen" % out_name)
print(phen_fpath, flush=True)

df_phen = pd.DataFrame({"IID": sample_ind,
              "FID": np.arange(0,N),
               "Y": y })
df_phen.to_csv(phen_fpath, index=False, header=None, sep="\t")

# Save true signals to file
print("\n...Saving true signals and probe names")
ts_fpath = os.path.join(out_dir, "%s_ts.csv" % out_name)
ts_bin_fpath = os.path.join(out_dir, "%s_ts.bin" % out_name)
probes_fpath = os.path.join(out_dir, "%s_probes.csv" % out_name)
print(ts_fpath, flush=True)
print(ts_bin_fpath, flush=True)
print(probes_fpath, flush=True)

ts_binf = open(ts_bin_fpath, "wb")
ts_binf.write(struct.pack(str(M)+'d', *beta.squeeze()))
ts_binf.close()

df_probes = pd.DataFrame({})
df_probes["PROBE"] = probe_names
df_probes["BETA"] = beta
df_probes.to_csv(ts_fpath, index=None, sep="\t")
df_probes["PROBE"].to_csv(probes_fpath, index=None, header=None, sep="\t")

# Save true covariate effects to file
print("\n...Saving true covariate effects to file")
tc_fpath = os.path.join(out_dir, "%s_tc.csv" % out_name)
print(tc_fpath, flush=True)
df_tc = pd.DataFrame({})
df_tc["COV"] = cov_names
df_tc["DELTA"] = delta
df_tc.to_csv(tc_fpath, index=None, sep="\t")
