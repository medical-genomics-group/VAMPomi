import zarr
import numpy as np
import argparse
import os
import struct
import random

# This script simulate iid phenotype on the top of real data

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-zarr", "--zarr", help = "Path to zarr files")
parser.add_argument("-out", "--out", help = "Output file path")
parser.add_argument("-phen", "--phen", help = "Phenotype name", default="sim")
parser.add_argument("-dataset", "--dataset", help = "Dataset name")
parser.add_argument("-h2", "--h2", help = "Heritability", default=0.8)
parser.add_argument("-lam", "--lam", help = "Sparsity (lambda)", default=0.01)
parser.add_argument("-run", "--run", help = "Run number", default=0)
parser.add_argument("-mask", "--mask", help = "Path to train test samples mask")
parser.add_argument("-M", "--M", help = "Number of markers")
parser.add_argument("-N", "--N", help = "Number of samples")
args = parser.parse_args()

# Input parameters
zarr_fpath = args.zarr
out_fpath = args.out
phen_name = args.phen
dataset_name = args.dataset
mask_fpath = args.mask
lam = float(args.lam)
h2 = float(args.h2)
run = int(args.run)
M = int(args.M) # number of markers
N = int(args.N) # number of samples

# Subscrit for simulation
sub = "h2_%d_lam_%d_run_%d" % (h2*100, lam*100, run)

# File names for simulation
fname = "%s_%s_%s" % (dataset_name, phen_name, sub)
fname_train = "%s_train_%s_%s" % (dataset_name, phen_name, sub)
fname_test = "%s_test_%s_%s" % (dataset_name, phen_name, sub)

# Load train test indicator variable
msk = np.loadtxt(mask_fpath).astype(bool)

# List of zarr files in directory 
files = os.listdir(zarr_fpath)

# Number of causal markers
cm = int(M * lam)

# beta variance
bvar = 1 / cm

# indices of causal markers
idx = random.sample(range(M), cm)

# true signals beta
beta = np.zeros(M)
beta[idx] = np.random.normal(0,np.sqrt(bvar),cm)
print("Var(beta) =", bvar)

# Save true signals to file
np.savetxt(os.path.join(out_fpath, fname + ".beta_true"), beta, fmt="%0.10f")
beta_true_binf = open(os.path.join(out_fpath, fname + "_beta_true.bin"), "wb")
beta_true_binf.write(struct.pack(str(M)+'d', *beta.squeeze()))
beta_true_binf.close()

# Vector for storing X @ beta
g = np.zeros(N)

# Total number of loaded markers
Mtot = 0

# For all chrommosomes
for i,f in enumerate(files):

    # zarr file name
    print("Processing file %s" % f)
    
    # Loaded chunk of data (one chrommosome)
    store = zarr.open(os.path.join(zarr_fpath, f))
    
    # Check dimensions
    Ni = np.shape(store)[0]
    Mi = np.shape(store)[1]
    if N != Ni:
        raise Exception("Number of samples in zarr file and specified do not mach!")

    # X @ beta for current chromosome
    g += np.matmul(np.array(store), beta[Mtot:(Mtot+Mi)])
    Mtot += Mi
    
    # free memmory
    del store
    
if Mtot != M:
    raise Exception("Number of markers in zarr files and specified do not mach!")   
print("Total number of loaded markers:", Mtot)

# Variance of noise based on specified heritability 
evar = 1.0 / h2 - 1.0

# Add noise to g
y = g + np.random.normal(0, np.sqrt(evar), N)

print("Var(g) =", np.var(g))
print("Var(y) =", np.var(y))
print("h2 =", np.var(g) / np.var(y))

# Standardize phenotype
y = (y - np.mean(y)) / np.std(y)
print("After standardization: Var(y) =", np.var(y))

# Open output files for phenotype
phenf_test = open(os.path.join(out_fpath, fname_test + ".phen"), "w")
phenf_train = open(os.path.join(out_fpath, fname_train + ".phen"), "w")

for i, pheno in enumerate(y):
    
    line = "%d %d %0.10f\n" % (i, i, pheno)
    
    # train test splits
    if msk[i]:
        phenf_train.write(line)
    else:
        phenf_test.write(line)

# Close output files
phenf_train.close()
phenf_test.close()