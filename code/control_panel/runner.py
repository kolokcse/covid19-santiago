import os
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from subprocess import Popen, STDOUT, PIPE

from logger import TBLogger

def moving_average(a, n=7) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def read_yaml(filename="input.yaml"):
    with open(filename) as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    return args

def run(c_args):
    str_args = [str(item) for pair in c_args.items() for item in pair]
    p = Popen([ "../bin/main"] + str_args,
          stdout=PIPE, stdin=PIPE, stderr=STDOUT, bufsize=1, universal_newlines=True)
    p.communicate()

def get_inf_curve(df, death_rate, K):
    # Works for only 2 I compartment
    # TODO: make it tiny
    inf_cols = [c for c in df.columns if c[0:2]=='I_']
    inf_cols2 = [c for c in df.columns if c[0:2]=='I2']
    Is = df.filter(inf_cols, axis=1)
    Is2 = df.filter(inf_cols2, axis=1)
    
    I = np.zeros((len(Is), K, len(Is.columns)//K))
    for c in Is.columns:
        _,city,age = c.split("_")
        I[:,int(age), int(city)] = Is.loc[:, c]
    I2 = np.zeros((len(Is2), K, len(Is2.columns)//K))
    for c in Is2.columns:
        _,city,age = c.split("_")
        I2[:,int(age), int(city)] = Is2.loc[:, c]
    
    I = np.sum(I, axis=2)
    I2 = np.sum(I2, axis=2)
    # Aggregate over cities
    return np.sum(I*death_rate, axis=1)+np.sum(I2*death_rate, axis=1), Is.sum(axis=1), Is2.sum(axis=1)

def aggregate_county(pop_file, df, K):
    # Get city indexes of county
    with open(pop_file) as file:
        rows = []
        for row in json.load(file)["populations"]:
            rows.append((row["index"], row["city"],row["admin_municip"], row["admin_county"]))
        city_df = pd.DataFrame(rows, columns=["index", "city", "municip", "county"])
        network_size = len(city_df)
        county_IDs = city_df.groupby('county').index.apply(list).to_dict()

    charts = []
    for county, cities in county_IDs.items():
        infs = [f"I_{city}_{age}" for city in cities for age in range(K)] + \
               [f"I2_{city}_{age}" for city in cities for age in range(K)]
        chart = df[infs].agg(sum, axis=1)
        charts.append((county, chart))
    
    df = pd.DataFrame({county:chart for county,chart in charts})
    df.to_csv("log/{args['sim_id']}/county.csv")
    return network_size, charts

def aggregate_age(df, K, network_size):
    charts = []
    for age in range(K):
        infs = [f"I_{city}_{age}" for city in range(network_size)] + \
               [f"I2_{city}_{age}" for city in range(network_size)]
        chart = df[infs].agg(sum, axis=1)
        charts.append(chart)
            
    df = pd.DataFrame({str(age):charts[age] for age in range(K)})
    df.to_csv("log/{args['sim_id']}/ages.csv")


def aggregate_all(df, K, network_size):
    # TODO multiply with death ratio
    infs = [f"I_{city}_{age}" for city in range(network_size) for age in range(K)] + \
            [f"I2_{city}_{age}" for city in range(network_size) for age in range(K)]
    chart = df[infs].agg(sum, axis=1)

if __name__ == "__main__":
    ########################
    #         ARGS         #
    ########################
    # === Script Options ===
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--yaml', dest='yaml', default="input.yaml", help='Input yaml file')
    parser.add_argument('--nosim', dest='sim', action='store_false',default=True, help='Do simulation, or use already simulated]')
    parser.add_argument('--args', dest='show_args', action='store_true',default=False, help='Shows args from .yaml')
    options = parser.parse_args()

    # === Read args ===
    args = read_yaml()
    if(options.show_args):
        print("[main] Args:")
        for key,val in args.items():
            print(f"         {key} : {str(val)}")

    # === Check some consistency in the doc ===
    assert(args['age_groups'] == len(args['death_rate']))
    assert(os.path.exists(args['network_config_folder']))

    ########################
    #      SIMULATION      #
    ########################
    if(not  os.path.exists(f"log/{args['sim_id']}")):
        os.mkdir(f"log/{args['sim_id']}")
    # === Run simulation ===
    c_args = {
        "--out": f"log/{args['sim_id']}/sim.csv",
        "--config": args['network_config_folder'],
        "--maxT": args['simulated_days'],
        "--R0": args['first_wave']['R0'],
        "--second_ratio": args['second_wave']['R0_ratio'],
        "--second_wave": args['second_wave']['time'],
        "--c": args['seasonality'],
    }
    if(options.sim):
        run(c_args)
        print('[main] Simulation ended')
    else:
        print('[main] Simulations skiped')
    
    ########################
    #       LOG/SIM        #
    ########################
    # === Read simulation data ===
    df = pd.read_csv(f"log/{args['sim_id']}/sim.csv")
    pop_file = f"{args['network_config_folder']}/populations_KSH.json"

    # LOG: Deaths
    deaths, I, I2 = get_inf_curve(df, args['death_rate'], args['age_groups'])

    # LOG: County infections
    network_size, c_charts = aggregate_county(pop_file, df, args['age_groups'])
    
    # LOG: Age groups
    aggregate_age(df, args['age_groups'], network_size)
    
    ########################
    #       LOG/LOSS       #
    ########################
    #city_data = pd.read_csv("../hun_codes/data/HU_settlement_tempinfo.csv")
    home = "../.."
    county_data = pd.read_csv(f"{home}/code/hun_codes/data/halalozas_megyenkent.csv").fillna(method='ffill')
    county_data=county_data.rename(lambda l: l if l!="Budapest" else "főváros")[county_data.columns[1:]].diff(axis=0).dropna()
    county_data[county_data<0]=0
    county_data = county_data.rolling(7).mean().dropna()

    # Megyénkénti log
    losses = []
    for i in range(80):
        equal_ratio = np.sum([chart for label,chart in c_charts])/np.sum(county_data.iloc[154-i:154+args['simulated_days']-i].fillna(0).values)
        c_loss_sum = 0
        for county,chart in c_charts:
            if(county == "főváros"): county="Budapest"
            if(county not in county_data.columns):
                print(county)
            else:
                g_truth = equal_ratio*county_data[county].to_numpy()[154-i:154+args['simulated_days']-i]
                loss = np.sum(np.abs(g_truth-chart)**1)
                c_loss_sum += loss/(args['simulated_days'])
        #print(c_loss_sum/(19))
        losses.append((c_loss_sum, equal_ratio))
    
    ind_min = np.argmin(losses)
    shift,equal_ratio =losses[ind_min]
    print(shift, equal_ratio)

    #county_data.reindex()...
    county_data.to_csv(f"log/{args['sim_id']}/country_loss.csv")
    


    #print(county_data[154:][["Budapest", "Dátum"]])
    # TODO:
    #    * Change plot to plotly
    #    * read real data
    #    * simple loss function