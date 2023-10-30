import numpy as np
from sklearn.metrics import r2_score
import argparse

# This script calculates R2 metric
print("...Calculating R2 metric", flush=True)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-est", "--est", help = "Path to y estimates file")
parser.add_argument("-true", "--true", help = "Path to true phen file")
args = parser.parse_args()
est_fpath = args.est
phen_fpath = args.true

print("Input arguments:")
print("--est", est_fpath)
print("--true", phen_fpath)
print("\n", flush=True)

# load data from file
def load_file(path, col):
    y = []
    file = open(path, "rb")
    for row in file:
        row = row.split()
        y.append(float(row[col]))
    return np.array(y)

# There is only one column in y_est file
y_est = load_file(est_fpath, 0)

# true phen file is in plink format
y_true = load_file(phen_fpath, 2)

# calculate R2
r2 = r2_score(y_true, y_est)

print("R2 = %0.4f" % r2, flush=True)