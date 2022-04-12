import os
import copy
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
from subprocess import Popen, STDOUT, PIPE

def run(args, job_count, lock):
    # Run
    str_args = [str(item) for pair in args.items() for item in pair]
    p = Popen([ "../bin/main"] + str_args,
          stdout=PIPE, stdin=PIPE, stderr=STDOUT, bufsize=1, universal_newlines=True)

    out,err = p.communicate()
    #print(out)
    
    # Alert master, if ready
    with lock:
        job_count[0]+=1
        print('\r {}/{}'.format(job_count[0], job_count[1]), end='', flush=True)

    return 0

def measure(meas_name, arg_name, arg_space):
    # Init pool
    pool = multiprocessing.Pool(processes=global_args["procnum"])
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    job_count = manager.Array("i", [0,len(arg_space)])
    
    os.makedirs(f"../output/{meas_name}", exist_ok=True)
    
    # Run async
    sims = {}
    for i,x in enumerate(arg_space):
        if type(x)==str:
            pass
        else:
            x = np.round_(x, decimals=4)
        args = copy.copy(base_args)
        args["--out"]=f"../output/{meas_name}/{i}.txt"
        args[arg_name]=f"{x}"
        sims[(x,args["--out"])] = pool.apply_async(run, args =(args, job_count, lock))
    
    # Collect async
    for k,v in sims.items():
        v.get()
    
    pool.close()
    pool.join()
    
    return sims.keys()

"""
def convert_ages(arr):
    # Split first age into 3
    ret = [arr[0], arr[0], arr[0]]
    # Then split the others into 2 category
    for e in arr[1:-1]:
        ret += [e,e]
    # Finally add the last
    ret += [arr[-1]]
    return ret

def get_inf_curve(filename, death = None):
    df = pd.read_csv(filename, sep=',')
    inf_cols = [c for c in df.columns if c[0]=='I']
    Is = df.filter(inf_cols, axis=1)
    
    I = np.zeros((150, 16, len(Is.columns)//16))
    for c in Is.columns:
        _,city,age = c.split("_")
        I[:,int(age), int(city)] = Is.loc[:, c]
    
    I = np.sum(I, axis=2)
    if type(death) != None:
        return np.sum(I*death, axis=1), Is.sum(axis=1)
    else:
        return Is.sum(axis=1)
"""

base_args = {
    "--config": "../input/hun",
    "--out": "../output/sim/temp.txt",
    "--maxT": "150",
    "--c": "0.25" # Seasonality
}

global_args = {
    "procnum": 20,
}

sims_R0 = measure("R0_K2", "--R0", np.linspace(1.5, 2.5, 20))