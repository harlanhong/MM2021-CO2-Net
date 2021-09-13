import pandas as pd
import pdb
import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import savgol_filter
import torch
import matplotlib as mlp

from matplotlib.pyplot import MultipleLocator
mlp.rcParams['axes.spines.right'] = False
mlp.rcParams['axes.spines.top'] = False
def branch_num_thumos():
    plt.grid(linestyle='-.')
    avg_data = [45.197 ,
                45.443 ,
                44.891 ,
                45.294 ,
                45.456 ,
                45.284 ,
                45.288 ,
                45.256 ]
    var_data = [0.049 ,
                0.019 ,
                0.366 ,
                0.035 ,
                0.347 ,
                0.072 ,
                0.091 ,
                0.084 ]
    avg_idx =  [2,
                4,
                6,
                8,
                10,
                12,
                16,
                20]
    x = list(range(len(avg_idx)))
    plt.errorbar(x,avg_data,yerr=var_data,fmt='o-g',lw = 2,ecolor='firebrick',elinewidth=3,ms=7,capsize=3)

    plt.xticks(np.arange(len(x)), avg_idx,fontsize=12)

    # plt.legend()  
    plt.savefig("./temp/hyperpa/branch_eval_thumos.pdf")  

    plt.savefig("./temp/hyperpa/branch_eval_thumos.jpg")  
    print('save')
    plt.clf()
def seq_len_thumos():
    plt.grid(linestyle='-.')
    # avg_data = [40.969,44.036 ,44.620 ,44.700 ,44.831 ,44.733 ,44.870 ,45.097 ,44.938 ,44.897]
    # var_data = [0.130,0.146 ,0.081 ,0.057 ,0.094 ,0.022 ,0.037 ,0.043 ,0.065 ,0.034 ]
    # avg_idx =  [120,180,240,280,360,400,480,560,640,800]
    avg_data = [44.831 ,
                45.150 ,
                45.298 ,
                45.313 ,
                45.067 ,
                45.286 ,
                45.443 ,
                45.177 ,
                44.992 
               ]
    var_data = [0.086 ,
                0.110 ,
                0.069 ,
                0.061 ,
                0.003 ,
                0.040 ,
                0.019 ,
                0.048 ,
                0.077 ]
    avg_idx =  [240,
                280,
                320,
                360,
                400,
                480,
                560,
                640,
                800]
    x = list(range(len(avg_idx)))
    plt.errorbar(x,avg_data,yerr=var_data,fmt='o-g',lw = 2,ecolor='firebrick',elinewidth=3,ms=7,capsize=3)

    plt.xticks(np.arange(len(x)), avg_idx,fontsize=12)

    # plt.legend()  
    plt.savefig("./temp/hyperpa/seq_len_eval_thumos.pdf")  

    plt.savefig("./temp/hyperpa/seq_len_eval_thumos.jpg")  
    print('save')
    plt.clf()

def topk_thumos():
    plt.grid(linestyle='-.')
    avg_data = [43.816 ,
                44.374 ,
                44.877 ,
                45.262 ,
                45.443 ,
                45.355 ,
                45.270 ,
                45.309 ]
    var_data = [0.065 ,
                0.053 ,
                0.017 ,
                0.038 ,
                0.019 ,
                0.087 ,
                0.110 ,
                0.083 ]
    avg_idx =  [3,
                4,
                5,
                6,
                7,
                8,
                9,
                10]
    x = list(range(len(avg_idx)))
    plt.errorbar(x,avg_data,yerr=var_data,fmt='o-g',lw = 2,ecolor='firebrick',elinewidth=3,ms=7,capsize=3)

    plt.xticks(np.arange(len(x)), avg_idx,fontsize=12)

    # plt.legend()  
    plt.savefig("./temp/hyperpa/topk_thumos.pdf")  

    plt.savefig("./temp/hyperpa/topk_thumos.jpg")  
    print('save')
    plt.clf()

def topk_ant():
    plt.grid(linestyle='-.')
    # avg_data = [40.969,44.036 ,44.620 ,44.700 ,44.831 ,44.733 ,44.870 ,45.097 ,44.938 ,44.897]
    # var_data = [0.130,0.146 ,0.081 ,0.057 ,0.094 ,0.022 ,0.037 ,0.043 ,0.065 ,0.034 ]
    # avg_idx =  [120,180,240,280,360,400,480,560,640,800]
    avg_data = [
        26.642 ,
        26.690 ,
        26.860,
        26.582 ,
        26.508 ,
        26.530 ,
        26.426 
    ]
    var_data = [
        0.018 ,
        0.006 ,
        0.001 ,
        0.008 ,
        0.005 ,
        0.018 ,
        0.013 
    ]
    avg_idx =  [
        3,
        4,
        5,
        6,
        7,
        8,
        9
    ]
    x = list(range(len(avg_idx)))
    plt.errorbar(x,avg_data,yerr=var_data,fmt='o-g',lw = 2,ecolor='firebrick',elinewidth=3,ms=7,capsize=3)

    plt.xticks(np.arange(len(x)), avg_idx,fontsize=12)

    # plt.legend()  
    plt.savefig("./temp/hyperpa/topk_ant.pdf")  

    plt.savefig("./temp/hyperpa/topk_ant.jpg")  
    print('save')
    plt.clf()


def branch_num_ant():
    plt.grid(linestyle='-.')
    avg_data = [26.581 ,
                26.554 ,
                26.627 ,
                26.653 ,
                26.560 ,
                26.668 ,
                26.609 ,
                26.664 ,
                26.654 ]
    var_data = [0.009 ,
                0.004 ,
                0.031 ,
                0.011 ,
                0.006 ,
                0.018 ,
                0.001 ,
                0.010 ,
                0.025 ]
    avg_idx =  [2,
                4,
                6,
                8,
                10,
                12,
                16,
                20,
                32]
    x = list(range(len(avg_idx)))
    plt.errorbar(x,avg_data,yerr=var_data,fmt='o-g',lw = 2,ecolor='firebrick',elinewidth=3,ms=7,capsize=3)

    plt.xticks(np.arange(len(x)), avg_idx,fontsize=12)

    # plt.legend()  
    plt.savefig("./temp/hyperpa/branch_eval_ant.pdf")  

    plt.savefig("./temp/hyperpa/branch_eval_ant.jpg")  
    print('save')
    plt.clf()

def seq_len_ant():
    plt.grid(linestyle='-.')
    avg_data = [26.435 ,
                26.609 ,
                25.974 ,
                25.543 ,
                24.866 ]
    var_data = [0.027 ,
                0.001 ,
                0.009 ,
                0.021 ,
                0.022 ]
    avg_idx =  [40,
                60,
                80,
                100,
                120]
    x = list(range(len(avg_idx)))
    plt.errorbar(x,avg_data,yerr=var_data,fmt='o-g',lw = 2,ecolor='firebrick',elinewidth=3,ms=7,capsize=3)

    plt.xticks(np.arange(len(x)), avg_idx,fontsize=12)

    # plt.legend()  
    plt.savefig("./temp/hyperpa/seq_len_eval_ant.pdf")  

    plt.savefig("./temp/hyperpa/seq_len_eval_ant.jpg")  
    print('save')
    plt.clf()

if __name__ == "__main__":
    branch_num_thumos()
    seq_len_thumos()
    topk_thumos()

    topk_ant()
    branch_num_ant()
    seq_len_ant()

