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
    pool = multiprocessing.Pool(processes=min(global_args["procnum"], len(arg_space)))
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

base_args = {
    "--config": "../input/hun",
    "--maxT": "160",
    "--R0":2.0,
    "--second_ratio":3.0,
    "--second_wave":80,
}

global_args = {
    "procnum": 20,
}

#sims_F = measure("second_wave/second_T1:80_R0:2.0", "--second_ratio", [3.0, 3.5, 4.0])
sims_R0 = measure("second_wave/second_T1:80_F:3.0", "--R0", np.linspace(2.0, 2.6, 10))