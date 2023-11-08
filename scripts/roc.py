import numpy as np
import argparse
import struct
import os
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt

# This script visualize ROC curve
print("...Computing ROC curve", flush=True)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-pip", "--pip", help = "Path to pip bin file")
parser.add_argument("-true_signal", "--true-signal", help = "Path to true signal binary file")
parser.add_argument("-M", "--M", help = "Number of markers")
parser.add_argument("-th", "--th", help = "PIP threshold", default=0.95)
args = parser.parse_args()

pipfile = args.pip
true_signal_file = args.true_signal
M = int(args.M)
pip_th = args.th

# Get basename of file from input pip file
basename = os.path.basename(pipfile)
basename = basename.split('.')[0]
dirpath = os.path.dirname(pipfile)

print("Input arguments:")
print("--pip", pipfile)
print("--true-signal", true_signal_file)
print("--M", M)
print("--th", pip_th)
print("\n", flush=True)

# load posterior inclusion probabilities 
f = open(pipfile, "rb")
buffer = f.read(M * 8)
pip = struct.unpack(str(M)+'d', buffer)
pip = np.array(pip)

# load true signals
f = open(true_signal_file, "rb")
buffer = f.read(M * 8)
beta = struct.unpack(str(M)+'d', buffer)
beta = np.array(beta)

# true pip vector. 1 where true signal is nonzero, otherwise 0
true = np.zeros(M)
true[np.abs(beta) > 0] = 1

# get roc curve
fpr, tpr, thresholds = roc_curve(true, pip)

# plot roc
plt.plot(fpr, tpr, label="GMRM PIP")
plt.plot(np.array([0, 1]), np.array([0, 1]), "k--")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig(os.path.join(dirpath, basename+'_roc.png'))

# threshold pip
est = np.zeros(M)
est[pip > pip_th] = 1

# compute FDR
tn, fp, fn, tp = confusion_matrix(true, est).ravel()
fdr = fp / (fp + tp)
nd = sum(pip > pip_th) # number of discoveries

print("FDR = %0.4f" % fdr)
print("Number of discoveries = %d" % nd, flush=True)