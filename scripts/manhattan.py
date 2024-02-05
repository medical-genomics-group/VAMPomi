import numpy as np
import argparse
import struct
import os
import matplotlib.pyplot as plt
import pandas as pd

# This script visualize manhattan plot
print("---------- Association testing for VAMPomi ----------")
print("\n", flush=True)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-pval", "--pval", help = "Path to pvals bin file")
parser.add_argument("-probes", "--probes", help = "Path to probes file")
parser.add_argument("-out_name", "--out-name", help = "Output file name")
parser.add_argument("-M", "--M", help = "Number of markers")
parser.add_argument("-th", "--th", help = "P-values threshold", default=0.05)
args = parser.parse_args()
pvalfile = args.pval
probes_fpath = args.probes
out_name = args.out_name
M = int(args.M)
th = float(args.th)

print("Input arguments:")
print("--pval", pvalfile)
print("--probes", probes_fpath)
print("--out-name", out_name)
print("--M", M)
print("\n", flush=True)

dirpath = os.path.dirname(pvalfile)

print("...Reading p-values from the file")
print(pvalfile)
print("\n", flush=True)

# Get probes
Mt = 0
Mchr = []
probes = []
chrs = []
for chr in range(22):
    probes_df = pd.read_csv(probes_fpath+str(chr+1)+".txt", header=None)
    probesj = probes_df[0].to_list()
    Mj = len(probesj)
    for p in probesj:
        probes.append(p)
        chrs.append(chr+1)
    Mt+=Mj
    Mchr.append(Mj)

if Mt != M:
    raise Exception("Number of markers specified %d is not same as in probes file %d!" % (M, Mt))

probes = np.array(probes)
chrs = np.array(chrs)

pval_th = th / M

plt.plot(np.array([0, M]), np.array([-np.log10(pval_th), -np.log10(pval_th)]), "k--")
plt.xlabel('Probes')
plt.ylabel('-log10(p-value)')
plt.title("VAMPomi marker discoveries")

f = open(pvalfile, "rb")
buffer = f.read(M * 8)
pvals = struct.unpack(str(M)+'d', buffer)
pvals = np.array(pvals)

for chr in range(22):
    js = sum(Mchr[:chr])
    je = sum(Mchr[:chr+1])
    pvals_chr = pvals[js:je]
    plt.scatter(x=np.arange(js,je), y=-np.log10(pvals_chr), s=1)

fout = os.path.join(dirpath, out_name+'.png')
print("...Saving manhattan figure to file")
print(fout)
print("\n", flush=True)
plt.savefig(fout)

print("-"*28)
print("| Number of causal markers |")
print("-"*28)
print("| %24d |" % sum(pvals <= pval_th))
print("-"*28)
print("\n", flush=True)