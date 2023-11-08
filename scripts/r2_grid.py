import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import csv

# parse log output file from gVAMP
def parse(txt, par, step):
    vals = []
    sub1s = txt.split("********************")
    for sub1 in sub1s:
        sub2s = sub1.split("______________________")
        for sub2 in sub2s:
            if step in sub2:
                lines = sub2.split("\n")
                for line in lines:
                    if par in line:
                        val = float(line.split("=")[1])
                        vals.append(val)
    return vals

# This script returns evaluation metrics for gVAMP grid search
print("...Colecting metrics for VAMP grid search")
print("\n", flush=True)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-csv", "--csv", help = "Path to csv file")
parser.add_argument("-csv_test", "--csv-test", help = "Path to test csv file")
parser.add_argument("-log", "--log", help = "Path to log file")
parser.add_argument("--conf", "--conf", help = "Path to configuration file")
parser.add_argument("-range", "--range", help = "Range of ids in grid serach (e.g. 0:17)")
parser.add_argument("-iterations", "--iterations", help = "Number of iterations", default=35)

# Parse arguments
args = parser.parse_args()
csvf = args.csv
csv_test = args.csv_test
logf = args.log
ran = args.range
conf_fpath = args.conf
iterations = int(args.iterations)

# Parse range of ids
id_start = int(ran.split(":")[0])
id_end = int(ran.split(":")[1])
n = (id_end - id_start) # total number of runs

print("Input arguments:")
print("--csv", csvf)
print("--csv-test", csv_test)
print("--log", logf)
print("--range", ran)
print("--conf", conf_fpath)
print("--iterations", iterations)
print("\n", flush=True)

# Get basename of train csv
csv_base = os.path.basename(csvf)
csv_base = csv_base.split('_id')[0]
csv_dirpath = os.path.dirname(csvf)

# Get basename of test csv
csv_test_base = os.path.basename(csv_test)
csv_test_base = csv_test_base.split('_id')[0]
csv_test_dirpath = os.path.dirname(csv_test)

# Get basename of log file
log_base = os.path.basename(logf)
log_base = log_base.split('_id')[0]
log_dirpath = os.path.dirname(logf)

# Load configuration file
conf_file = open(conf_fpath)
conf_reader = csv.reader(conf_file, delimiter='\t', )
next(conf_reader, None) # skip header
rhos = []
Ls = []
lams = []

for row in conf_reader:
    rhos.append(float(row[1]))
    Ls.append(int(row[2]))
    lams.append(float(row[3]))

# Figures
fig1, ax1 = plt.subplots(3, figsize=(12, 8), dpi=200) # Test R2
fig2, ax2 = plt.subplots(3, figsize=(12, 8), dpi=200) # Train R2
fig3, ax3 = plt.subplots(3, figsize=(12, 8), dpi=200) # gam1
fig4, ax4 = plt.subplots(3, figsize=(12, 8), dpi=200) # gamw
fig5, ax5 = plt.subplots(3, figsize=(12, 8), dpi=200) # signal correlation

fig1.suptitle("Test R2")
fig2.suptitle("gam1")
fig3.suptitle("gamw")
fig4.suptitle("Train R2")
fig5.suptitle("Correlation xhat1")

# iterate over all runs
for r in range(id_start, id_end+1):

    # get paths to current files 
    csv_fpath = os.path.join(csv_dirpath, csv_base + "_id_%d.csv" % r)
    csv_test_fpath = os.path.join(csv_test_dirpath, csv_test_base + "_id_%d.csv" % r)
    log_fpath = os.path.join(log_dirpath, log_base + "_id_%d.log" % r)
    #print("...processing files:")
    #print(csv_fpath)
    #print(csv_test_fpath, flush=True)

    # Load output csv files that store parameteres from train and test runs over iterations
    R2_test = []
    R2_denoising = []
    R2_lmmse = []
    gamw = []
    gam1 = []

    csv_file = open(csv_fpath, newline='', encoding='utf-8')
    # repleace nul values
    csv_reader = csv.reader((row.replace('\0', '') for row in csv_file), delimiter=',')
    
    for row in csv_reader:
        # train output csv file structure: 
        #| iteration | R2 denoising | L2 error denoising | R2 lmmse | L2 error lmmse | gamw | gam1 | gam2 | eta1 | eta2 | number of mixture components L | L * mixture probabilities | L * mixture variances |
        gam1.append(float(row[6]))
        gamw.append(float(row[5]))
        R2_denoising.append(float(row[1]))
        R2_lmmse.append(float(row[3]))

    csv_test_file = open(csv_test_fpath, newline='', encoding='utf-8')
    csv_reader = csv.reader((row.replace('\0', '') for row in csv_test_file),delimiter=',')
    
    for row in csv_reader:
        # test output csv file structure:
        # | iteration | R2_test |
        R2_test.append(float(row[1])) 

    R2_test = np.array(R2_test)
    R2_denoising = np.array(R2_denoising)
    R2_lmmse = np.array(R2_lmmse)
    gamw = np.array(gamw)
    gam1 = np.array(gam1)
    
    # calculate difference between R2 in denoising and lmmse step
    R2_diff = R2_lmmse - R2_denoising
    it = np.where(R2_diff < 0.05)[0][0] # get first iterations where lmmse R2 is close to denoising R2
    #it = np.argmax(gam1) # get iteration by largest gam1
    #it = np.argmax(R2_denoising) # get iteration by largest train R2

    # load log file
    log_file = open(log_fpath, "r")
    log_text = log_file.read()
    corr_xhat1 = parse(log_text, "correlation x1_hat", "DENOISING")

    # check if metrics across all iterations where loaded
    if (len(R2_test) != iterations) or (len(gamw) != iterations) or (len(gam1) != iterations) or (len(corr_xhat1) != iterations):
        print("len(R2)=%d, len(gamw)=%d, len(gam1)=%d, len(corr_xhat1)=%d" % (len(R2_test), len(gamw), len(gam1), len(corr_xhat1)))
        raise Exception("Missing iterations in ID %d" % r)

    #print("Max gam1 is in iteration:", it + 1)
    #print("heritability in iteration", it + 1, ": ", 1 - 1 / gamw[it])
    #print("Test R2 in iteration", it + 1, ": ", R2_test[it])
    #print("\n", flush=True)
    print("%d & %0.2f & %d & %0.3f & %0.4f & %0.4f & %0.4f & %0.4f & %d \\\\" % (r, rhos[r], Ls[r], lams[r], R2_test[it], gam1[it], gamw[it], corr_xhat1[it], it), flush=True)

    # Split to three subplots by damping factor rho
    axid = 0
    if rhos[r] == 0.05:
        axid = 0
    elif rhos[r] == 0.1:
        axid = 1
    else:
        axid = 2

    # Curve label
    lbl = "ID=%d, rho=%0.2f, L=%d, lam=%0.3f" % (r, rhos[r], Ls[r], lams[r])

    # plot test R2
    ax1[axid].plot(R2_test, label=lbl)
    ax1[axid].set_title("rho=%0.2f" % rhos[r])
    ax1[axid].set_ylim([0,0.3])
    ax1[axid].set_xlabel("Iteration")
    ax1[axid].set_ylabel("R2")
    ax1[axid].legend(loc="right", bbox_to_anchor=(1, 0.5))
    ax1[axid].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # plot train R2
    curves = ax4[axid].plot(R2_denoising, label=lbl)
    color = curves[0].get_color()
    ax4[axid].plot(R2_lmmse, linestyle='--', color=color)
    ax4[axid].set_title("rho=%0.2f" % rhos[r])
    ax4[axid].set_ylim([0,1])
    ax4[axid].set_xlabel("Iteration")
    ax4[axid].set_ylabel("R2")
    ax4[axid].legend(loc="right", bbox_to_anchor=(1, 0.5))
    ax4[axid].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # plot gam1
    ax2[axid].plot(gam1, label=lbl)
    ax2[axid].set_title("rho=%0.2f" % rhos[r])
    #ax2[axid].set_ylim([0,0.3])
    ax2[axid].set_xlabel("Iteration")
    ax2[axid].set_ylabel("gam1")
    ax2[axid].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # plot gamw
    ax3[axid].plot(gamw, label=lbl)
    ax3[axid].set_title("rho=%0.2f" % rhos[r])
    ax3[axid].set_xlabel("Iteration")
    ax3[axid].set_ylabel("gamw")
    ax3[axid].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # plot correlation
    ax5[axid].plot(corr_xhat1, label=lbl)
    ax5[axid].set_title("rho=%0.2f" % rhos[r])
    ax5[axid].set_ylim([0,0.3])
    ax5[axid].set_xlabel("Iteration")
    ax5[axid].set_ylabel("Correlation")
    ax5[axid].legend(loc="right", bbox_to_anchor=(1, 0.5))
    ax5[axid].legend(loc="center left", bbox_to_anchor=(1, 0.5))

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()

# Save figures
print("...saving figures to files")
fig1_path = os.path.join(csv_dirpath, csv_base+'_r2.png')
print(fig1_path, flush=True)
fig1.savefig(fig1_path)

fig2_path = os.path.join(csv_dirpath, csv_base+'_gam1.png')
print(fig2_path, flush=True)
fig2.savefig(fig2_path)

fig3_path = os.path.join(csv_dirpath, csv_base+'_gamw.png')
print(fig3_path, flush=True)
fig3.savefig(fig3_path)

fig4_path = os.path.join(csv_dirpath, csv_base+'_r2train.png')
print(fig4_path, flush=True)
fig4.savefig(fig4_path)

fig5_path = os.path.join(csv_dirpath, csv_base+'_corr.png')
print(fig5_path, flush=True)
fig5.savefig(fig5_path)