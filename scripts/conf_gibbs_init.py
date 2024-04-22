import numpy as np
import argparse
import csv
import os

def get_probs(L, lam):
    probs = [1-lam]
    while len(probs) <= (L-1):
        prob = np.round(1 - sum(probs), 10)
        if len(probs) == (L - 1): # this is the last probability
            probs.append(prob)
        else: 
            probs.append(prob / 2)

    if np.round(sum(probs), 10) != 1:
        raise Exception("Sum of probs should be 1!")

    probs_string = '%0.10f' % probs[0]
    for p in probs[1:]:
        probs_string += ',%0.10f' % p
    
    return probs, probs_string

def get_vars(L, var_max=0.1):
    vars = [0]
    var = (10 * var_max) / (10 ** (L-1))
    while len(vars) <= (L-1):
        vars.append(var)
        var = var * 10
    vars_str = '%0.12f' % vars[0]
    for v in vars[1:]:
        vars_str += ',%0.12f' % v

    return vars, vars_str

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-csv", "--csv", help = "Path to csv file from GMRMomi")
parser.add_argument("-grm", "--grm", help = "Path to group mixtures file", default="")
parser.add_argument("-out_dir", "--out-dir", help = "Path to output directory", default="")
parser.add_argument("-iterations", "--iterations", help = "Iterations range start:end", default="100:200")
parser.add_argument("-rho", "--rho", help = "Damping factor", default=0.5)

args = parser.parse_args()

fcsv = args.csv
fgrm = args.grm
ran = args.iterations
out_dir = args.out_dir
rho = float(args.rho)

start = int(ran.split(':')[0])
end = int(ran.split(':')[1])

# Get basename of file from input csv file
base = os.path.basename(fcsv)
base = base.split('.')[0]
dirpath = os.path.dirname(fcsv)

sigmag = []
sigmae = []
h2 = []
mincl = []
L = 0
probs = []
with open(fcsv) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        #print(row)
        sigmag.append(float(row[2]))
        sigmae.append(float(row[3]))
        h2.append(float(row[4]))
        mincl.append(float(row[5]))
        L = int(row[7])
        p = np.zeros(L)
        for i in range(L):
            p[i] = float(row[8+i])
        probs.append(p)

sigmag = np.array(sigmag[start:end])
sigmae = np.array(sigmae[start:end])
h2 = np.array(h2[start:end])
mincl = np.array(mincl[start:end])
probs = np.array(probs[start:end])

f = open(fgrm,"r")
grm = f.readline()
vars = [float(m) for m in grm.split(' ')]
vars_str = "%0.12f" % vars[0]
for v in vars[1:]:
    vars_str += ',%0.12f' % v

prob_means = np.mean(probs, axis=0)
lam = (1 - prob_means[0])
h2_mean = np.mean(h2)

probs_str = "%0.12f" % prob_means[0]
for p in prob_means[1:]:
    probs_str += ',%0.12f' % p

print("h2 = %0.4f" % h2_mean)
print("Incl. markers = %d" % np.mean(mincl))
print("lam = %0.4f" % lam)

fout = os.path.join(out_dir, base + ".conf")
f = open(fout, 'w', newline='')
writer = csv.writer(f, delimiter='\t')
writer.writerow(['ID', 'rho', 'mix_comp', 'lambda', 'probs', 'vars', 'h2'])
writer.writerow([0, rho, L, lam, probs_str, vars_str, h2_mean])