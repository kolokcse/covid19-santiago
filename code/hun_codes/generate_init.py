import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools


sett_types = pd.read_csv("../../data/hun/HU_places_admin_pop_ZIP_latlon.csv",
           sep=',',
           header=0)
KSH = pd.read_csv("../../data/hun/KSHCommuting_c1ID_c1name_c2ID_c2name_comm_school_work_DIR.csv",
           sep=',',
           header=0)

korfa_teltip1 = {
    "főváros": [210640, 81626, 246127, 313540, 213367, 225379, 216828, 221533, 1729040],
    "fővárosi kerület": [210640, 81626, 246127, 313540, 213367, 225379, 216828, 221533, 1729040],
    "megyeszékhely-megyei jogú város": [238608 , 116868 , 233585 , 287108 , 227042 , 252073 , 209260 , 197113 , 1761657],
    "megyei jogú város": [37141, 15599, 32610, 44865, 35886, 39168, 32719, 29748, 267736],
    "város": [477932, 195704, 364021, 491615, 422355, 466796, 373198, 353804, 3145425],
    "all_város": [964321, 409797, 876343, 1137128, 898650, 983416, 832005, 802198, 6903858],
    "község": [483338, 183737, 353193, 443785, 417543, 455266, 344957, 351951, 3033770],
    "nagyközség": [483338, 183737, 353193, 443785, 417543, 455266, 344957, 351951, 3033770],
    "all": [1447659 , 593534 , 1229536 , 1580913 , 1316193 , 1438682 , 1176962 , 1154149 , 9937628],
}
ages = ["0–14", "15–19", "20–29", "30–39", "40–49", "50–59", "60–69", "70–", "all"]

# === Populations and infected ===
def get_age(row, korfa_teltip, I, L, Ks):
    arr = np.array(korfa_teltip[row["settlement type"]])
    arr = arr[:-1]/np.sum(arr[:-1])
    N = row["population"]
    
    ages = (np.array(arr)*N).astype(int)
    
    corrected_ages = np.array([a if i in Ks else 0 for i,a in enumerate(ages)])
    Is = np.random.multinomial(I, corrected_ages/np.sum(corrected_ages))
    Ls = np.random.multinomial(L, corrected_ages/np.sum(corrected_ages))
    
    ages = [{"N": int(s), "S":int(s-i-l), "L":int(l), "I":int(i), "R":0} for s,l,i in zip(ages,Ls,Is)]
    return ages

def create_population_dict(sett_types, population_th, num_I, num_L, Ks, budapest):
    # === Restrict to small cities ===
    small_cities = sett_types[sett_types["population"]>population_th]
    small_cities = small_cities[small_cities["place"] != "Budapest"]

    pop = np.array(list(small_cities["population"].array))
    if budapest:
        for i,c in enumerate(small_cities["place"].array):
            if c[:8] != "Budapest":
                pop[i] = 0 
    
    # === Spread infected agents ===
    Is = np.random.multinomial(num_I, pop/np.sum(pop))
    Ls = np.random.multinomial(num_L, pop/np.sum(pop))
    
    place_id_dict = {}
    population = {"populations":[]}
    for ind, (_, row) in enumerate(small_cities.iterrows()):
        city = {
            "name": str(ind),
            "city":row["place"],
            #"name": row["place"],
            "index": ind,
            "N": row["population"],
            "r1":1.0,
            "r2":1.0,
            "age": get_age(row, korfa_teltip1, Is[ind], Ls[ind], Ks)
        }
        place_id_dict[row['place']] = ind
        population["populations"].append(city)
    
    with open("../input/hun/populations_KSH.json", "w") as f:
        f.write(json.dumps(population))
    return population, set(small_cities['place']), place_id_dict

# === Commuting ===
def create_commuting(cities):
    N = len(cities)
    pop_dict = {row['place']:row["population"] for _,row in sett_types.iterrows()}

    edges = {}
    mtx = np.zeros((N,N))
    for _,row in KSH.iterrows():
        weight = row["CommutersAll"]
        orig,dest = row["origName"], row["destName"]
        if((orig in cities) and (dest in cities)):
            edges[(place_id_dict[orig], place_id_dict[dest])] = weight/pop_dict[orig]
            mtx[place_id_dict[orig], place_id_dict[dest]] = weight/pop_dict[orig]
    

    network = [{"from":i, "to":j, "weight":mtx[i,j]} for i,j in itertools.product(range(N), range(N)) if i!= j]
    commuting = {
        "N":N,
        "network":network
    }

    with open("../input/hun/commuting_KSH.json", "w") as f:
        f.write(json.dumps(commuting))

# === Create contats_home and contacts other ===
import itertools

def select_specified_ages(mtx):
    indexes = [[0,2], [3,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15]]
    N = len(indexes)
    new = np.zeros((N,N))

    for i,j in itertools.product(range(N), range(N)):
        a,b = indexes[i]
        c,d = indexes[j]
        new[i,j] = np.mean(mtx[a:b+1,c:d+1])
    return new

def write_mtx(mtx, name):
    N = len(mtx)
    d = {
        "K":len(mtx),
        "rates": [{"from":i,"to":j, 'rate':mtx[i,j]} for i,j in itertools.product(range(N), range(N))]
    }
    with open(f"../input/hun/contacts_{name}.json", "w") as f:
        f.write(json.dumps(d))

def create_new_contact_mtx():
    d = json.load(open("../input/santiago/contacts_home.json"))
    K = int(d['K'])
    mtx_home = np.zeros((K,K))
    for d1 in d['rates']:
        mtx_home[d1["from"]][d1["to"]] = d1['rate']
        
    d = json.load(open("../input/santiago/contacts_other.json"))
    K = int(d['K'])
    mtx_other = np.zeros((K,K))
    for d1 in d['rates']:
        mtx_other[d1["from"]][d1["to"]] = d1['rate']
    
    new_home = select_specified_ages(mtx_home)
    new_other = select_specified_ages(mtx_other)
    write_mtx(new_home, "home")
    write_mtx(new_other, "other")

# === MAIN ===
pops,cities,place_id_dict = create_population_dict(
    sett_types, population_th=10000,
    num_I =10000,
    num_L=int(10000*(4/2.5)),
    Ks = [4,5],
    budapest = False)

print(f"Number of cities: {len(cities)}")
print(f'Number of age groups {len(pops["populations"][1]["age"])}')

create_commuting(cities)
create_new_contact_mtx()
#print(cities)    



