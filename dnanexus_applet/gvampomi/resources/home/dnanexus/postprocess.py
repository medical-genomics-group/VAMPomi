import numpy as np
import argparse
import struct
import pandas as pd
from scipy.stats import norm
import re

def pip_calc(r1, gam1, omegas, sigmas, la):
    r1 = np.asmatrix(r1)
    gam1inv = 1.0/gam1
    beta_tilde=np.multiply( np.exp( - np.power(np.transpose(r1),2) / 2 / (sigmas + gam1inv)), omegas / np.sqrt(gam1inv + sigmas) )
    sum_beta_tilde = beta_tilde.sum(axis=1)
    pi = la / ( la + (1-la) * np.exp(-np.power(np.transpose(r1),2) / 2 * gam1 ) / np.sqrt(gam1inv) / sum_beta_tilde )
    return np.asarray(pi).squeeze()

def pvals_calc(r1, gam1):
    r1 = np.asmatrix(r1)
    pvals = norm.cdf(-abs(r1), loc=0, scale=1/np.sqrt(gam1)) # one-sided test
    return np.asarray(pvals).squeeze()

def parse_prior(txt, par, step):
    vals = []
    sub1s = txt.split("********************")
    for sub1 in sub1s:
        sub2s = sub1.split("______________________")
        for sub2 in sub2s:
            if step in sub2:
                lines = sub2.split("\n")
                for line in lines:
                    if line.startswith(par):
                        val = (line.split("=")[1])
                        # p = re.compile(r'\d+\.\d+')  # Compile a pattern to capture float values
                        p = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
                        floats = [float(i) for i in p.findall(val)]  # Convert strings to float
                        vals.append(floats)
    return vals

def parse(txt, par, step):
    vals = []
    sub1s = txt.split("********************")
    for sub1 in sub1s:
        sub2s = sub1.split("______________________")
        for sub2 in sub2s:
            if step in sub2:
                lines = sub2.split("\n")
                for line in lines:
                    if line.startswith(par):
                        val = float(line.split("=")[1])
                        vals.append(val)
                        break
    return vals

# This script for gVAMP postprocessing
print("---------- gVAMPomi postprocessing ----------")
print("\n", flush=True)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-probes", "--probes", help = "Path to probes file")
parser.add_argument("-log", "--log", help = "Path to log file")
parser.add_argument("-params", "--params", help = "Path to params file")
parser.add_argument("-out_name", "--out-name", help = "Output file name")
parser.add_argument("-M", "--M", help = "Number of markers")
parser.add_argument("-N", "--N", help = "Number of samples")
args = parser.parse_args()

probes_fpath = args.probes
log_fpath = args.log
params_fpath = args.params
out_name = args.out_name
M = int(args.M)
N = int(args.N)

print("Input arguments:")
print("--probes", probes_fpath)
print("--log", log_fpath)
print("--params", params_fpath)
print("--out-name", out_name)
print("--M", M)
print("--N", N)
print("\n", flush=True)

print("...Reading log file",flush=True)
f = open(log_fpath, "r")
text = f.read()
f.close()

gamws = parse(text, "gamw", "LMMSE")
gam1s = parse(text, "gam1", "LMMSE")
it = len(gam1s)
gam1 = gam1s[it - 2]
if len(gamws) > 0:
    gamw = gamws[it - 1]
else:
    gamw = np.inf
vars = parse_prior(text, "Prior variances", "DENOISING")[it - 2]
probs = parse_prior(text, "Prior probabilities", "DENOISING")[it - 2]
L = len(vars)

print("...Reading r1 from the file",flush=True)
r1_fpath = f"estimates/{out_name}_r1_it_{it}.bin"
print(r1_fpath)
f = open(r1_fpath, "rb")
buffer = f.read(M * 8)
r1 = struct.unpack(str(M)+'d', buffer)
r1 = np.array(r1)

print("...Reading estimates from the file",flush=True)
xhat_fpath = f"estimates/{out_name}_it_{it}.bin"
print(xhat_fpath)
f = open(xhat_fpath, "rb")
buffer = f.read(M * 8)
xhat = struct.unpack(str(M)+'d', buffer)
xhat = np.array(xhat)

print("...Reading probes file",flush=True)
df_probe = pd.read_table(probes_fpath, header=None, names=['PROBE'])

sigmag = 1 - (1 / gamw)

print("Prior probs =", probs)
print("Prior vars =", vars)
print("L =", L)
print("gam1 =", gam1)
print("gamw =", gamw)
print("sigmaG =", sigmag)
print("it =", it)

print("Calculating PIPs in iteration", it, flush=True)

sigmas = np.array(vars[1:])
omegas = np.array([ p / sum(probs[1:]) for p in probs[1:]])
la = 1 - probs[0]

pips = pip_calc(r1, gam1 * N, omegas, sigmas, la)
pvals = pvals_calc(r1, gam1 * N)

n_assoc = sum(pips >= 0.95)

df = pd.DataFrame({ 'PROBE': df_probe['PROBE'],
                    'BETA': xhat,
                    'P': pvals,
                    'PIP': pips })

print("...Saving to file")
out_fpath = out_name + ".gvamp"
print(out_fpath)
print("\n", flush=True)
df.to_csv(out_fpath, index=None, sep="\t")

df_sum = pd.DataFrame({"sigmag": sigmag, "n_assoc": n_assoc}, index=[0])
df_sum.to_csv(out_name + "_summary.csv", index=False, sep="\t")