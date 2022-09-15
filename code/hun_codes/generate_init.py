import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from scipy import linalg

# === DATA ===
sett_types = pd.read_csv("../../data/hun/HU_places_admin_pop_ZIP_latlon.csv",
           sep=',',
           header=0)
KSH = pd.read_csv("../../data/hun/KSHCommuting_c1ID_c1name_c2ID_c2name_comm_school_work_DIR.csv",
           sep=',', header=0)
#KSH = pd.read_csv("../../data/hun/KSHCommuting_c1ID_c1name_c2ID_c2name_comm_school_work_UNDIR.csv",
#           sep=',', header=0)

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

def create_population_dict(sett_types, population_th, num_I, num_L, Ks, budapest, out):
    # === Restrict to small cities ===
    big_cities = sett_types[sett_types["population"]>population_th]
    big_cities = big_cities[big_cities["place"] != "Budapest"]

    pop = np.array(list(big_cities["population"].array))
    if budapest:
        for i,c in enumerate(big_cities["place"].array):
            if c[:8] != "Budapest":
                pop[i] = 0 
    
    # === Spread infected agents ===
    Is = np.random.multinomial(num_I, pop/np.sum(pop))
    Ls = np.random.multinomial(num_L, pop/np.sum(pop))
    
    place_id_dict = {}
    population = {"populations":[]}
    for ind, (_, row) in enumerate(big_cities.iterrows()):
        city = {
            "name": str(ind),
            #"name": row["place"],
            "city":row["place"],
            "index": ind,
            "N": row["population"],
            "admin_municip":row["admin municip"],
            "admin_county":row["admin county"],
            "r1":1.0,
            "r2":1.0,
            "age": get_age(row, korfa_teltip1, Is[ind], Ls[ind], Ks)
        }
        place_id_dict[row['place']] = ind
        population["populations"].append(city)
    
    with open(f"../input/{out}/populations_KSH.json", "w") as f:
        f.write(json.dumps(population))
    
    district_pops, district_id_dict = generate_district_pop_dict(population, big_cities, out=f"{out}")
    
    return population, place_id_dict, big_cities, district_pops, district_id_dict

def add_ages(ages1, ages2):
    ages = []

    for a1,a2 in zip(ages1, ages2):
        ages.append({"N":a1["N"]+a2["N"], "S":a1["S"]+a2["S"], "L":a1["L"]+a2["L"], "I":a1["I"]+a2["I"], "R":a1["R"]+a2["R"]})
    return ages

def generate_district_pop_dict(populations, big_cities, out):
    all_district = set(big_cities["admin municip"])
    get_district = big_cities.set_index('place').to_dict()["admin municip"]

    pops = {}
    district_id_dict = {}
    for ind,d in enumerate(all_district):
        district_id_dict[d]=ind
        dist = {
            "name":str(ind),
            "district":d,
            "index":ind,
            "N":0,
            "r1":1.0,
            "r2":1.0,
            "age": None
        }
        pops[d]=dist
    
    for city in populations["populations"]:
        dist_name = get_district[city["city"]]
        pops[dist_name]["N"] += city["N"]
        if pops[dist_name]["age"] == None:
            pops[dist_name]["age"] = city["age"]
        else:
            pops[dist_name]["age"] = add_ages(pops[dist_name]["age"], city["age"])

    with open(f"../input/{out}/district/populations_KSH.json", "w") as f:
        f.write(json.dumps({"populations":list(pops.values())}))
    
    with open(f"../input/{out}/district_eigen/populations_KSH.json", "w") as f:
        f.write(json.dumps({"populations":list(pops.values())}))
    
    return pops, district_id_dict

def get_eigen_mtx(mtx, district_ind, pop, out):
    """
    Description:
        Returns the eig.vals enhanced district aggregated mtx
    Parameters:
        * mtx          : commutation mtx between cities
        * district_ind : district dictionary, containing the cities in the district
        * pop          : population of each city
    """
    A = np.array([mtx[:,j]*pop[j] for j in range(len(mtx))])
    #phi = np.linalg.eigvals(A)
    eigvals, eigenVectors = np.linalg.eig(A)
    phi = np.real(eigenVectors[:,0])
    phi = phi/np.sum(phi)
    print("Largest eigenvalue:", eigvals[0], "gap %", 100*eigvals[1]/eigvals[0])
    print(sorted(eigvals, reverse=True))
    #print(list(phi))
    with open(f"../input/{out}/{th}.npy", "wb")as file:
        np.save(file, np.array(phi))
    
    
    mtx_d = np.zeros((len(district_ind), len(district_ind)))
    for l,k in itertools.product(district_ind.keys(), district_ind.keys()):
        up = 0
        for i,j in itertools.product(district_ind[l], district_ind[k]):
            up +=  mtx[i,j]*phi[j]
            #up +=  mtx[i,j]*phi[j]*pop[i]*pop[j]
        
        down = sum([phi[j]*pop[j] for j in district_ind[j]])
        mk = sum([pop[i] for i in district_ind[k]])
        ml = sum([pop[i] for i in district_ind[l]])
        
        if(down < 1e-12): mtx_d[l,k] = 0.0
        else: mtx_d[l,k] = ml*up/down
    
    return mtx_d

def get_eigen_mtx2(mtx, district_ind, pop, out):
    """
    Description:
        Returns the eig.vals enhanced district aggregated mtx
    Parameters:
        * mtx          : commutation mtx between cities
        * district_ind : district dictionary, containing the cities in the district
        * pop          : population of each city
    """
    A = np.array([mtx[:,j] for j in range(len(mtx))])

    eigvals, eigenVectors = np.linalg.eig(A)
    phi = np.real(eigenVectors[:,0])
    phi = phi/np.sum(phi)
    print("Largest eigenvalue:", eigvals[0], "gap %", 100*eigvals[1]/eigvals[0])
    print(sorted(eigvals, reverse=True))
    with open(f"../input/{out}/{th}.npy", "wb")as file:
        np.save(file, np.array(phi))
    
    
    mtx_d = np.zeros((len(district_ind), len(district_ind)))
    for l,k in itertools.product(district_ind.keys(), district_ind.keys()):
        up = 0
        for i,j in itertools.product(district_ind[l], district_ind[k]):
            up +=  mtx[i,j]*phi[j]
        
        down = sum([phi[j]*pop[j] for j in district_ind[j]])
        mk = sum([pop[i] for i in district_ind[k]])
        ml = sum([pop[i] for i in district_ind[l]])
        
        if(down < 1e-12): mtx_d[l,k] = 0.0
        else: mtx_d[l,k] = up/(len(district_ind[k])*np.sum([phi[j] for j in district_ind[k]]))
    
    return mtx_d

# === Commuting ===
def create_commuting(cities, place_id_dict, big_cities, district_pops, district_id_dict, out):
    N = len(cities)
    M = len(district_id_dict)
    pop_dict = {row['place']:row["population"] for _,row in sett_types.iterrows()}


    get_district = big_cities.set_index('place').to_dict()["admin municip"]

    mtx = np.zeros((N,N))
    mtx_dist = np.zeros((M,M))
    for _,row in KSH.iterrows():
        weight = row["CommutersAll"]
        orig,dest = row["origName"], row["destName"]
        if((orig in cities) and (dest in cities)):
            mtx[place_id_dict[orig], place_id_dict[dest]] += weight/pop_dict[orig]
            mtx[place_id_dict[dest], place_id_dict[orig]] += weight/pop_dict[dest]
            a = district_id_dict[get_district[orig]]
            b = district_id_dict[get_district[dest]]
            mtx_dist[a,b] += weight
            mtx_dist[b,a] += weight
    
    for d,ind in district_id_dict.items():
        mtx_dist[ind]/= district_pops[d]["N"]
    
    # === CITY COMMUTATION ===
    network = [{"from":i, "to":j, "weight":mtx[i,j]} for i,j in itertools.product(range(N), range(N)) if i!= j]
    commuting = {
        "N":N,
        "network":network
    }

    # === DISTRICT COMMUTATION ===
    print("Districts: ",M)
    network_dist = [{"from":i, "to":j, "weight":mtx_dist[i,j]} for i,j in itertools.product(range(M), range(M)) if i!= j]
    commuting_dist = {
        "N":M,
        "network":network_dist
    }

    # === EIGEN DISTRICT COMMUTATION ===
    district_ind = {i:[] for i in range(len(mtx_dist))}
    pop = np.zeros(len(mtx))
    for city in cities:
        pop[place_id_dict[city]] = pop_dict[city]
        
        act_dist = district_id_dict[get_district[city]]
        district_ind[act_dist].append(act_dist)
    
    city_names = ["" for i in range(N)]
    for city in place_id_dict:
        city_names[place_id_dict[city]] = city
    #print(city_names)
    with open(f"../input/{out}/{th}_cities.npy", "wb")as file:
        np.save(file, np.array(city_names))
    
    mtx_eigen = get_eigen_mtx(mtx, district_ind, pop, out)
    network_eigen = [{"from":i, "to":j, "weight":mtx_eigen[i,j]} for i,j in itertools.product(range(M), range(M)) if i!= j]
    network_eigen_symm = [{"from":i, "to":j, "weight":(mtx_eigen[i,j]+mtx_eigen[j,i])/2} for i,j in itertools.product(range(M), range(M)) if i!= j]

    with open(f"../input/{out}/commuting_KSH.json", "w") as f:
        f.write(json.dumps(commuting))
    
    with open(f"../input/{out}/district/commuting_KSH.json", "w") as f:
        f.write(json.dumps(commuting_dist))
    
    with open(f"../input/{out}/district_eigen/commuting_KSH.json", "w") as f:
        f.write(json.dumps({"N":M, "network":network_eigen}))

    with open(f"../input/{out}/district_eigen_symm/commuting_KSH.json", "w") as f:
        f.write(json.dumps({"N":M, "network":network_eigen_symm}))

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

def write_mtx(mtx, name, out):
    N = len(mtx)
    d = {
        "K":len(mtx),
        "rates": [{"from":i,"to":j, 'rate':mtx[i,j]} for i,j in itertools.product(range(N), range(N))]
    }
    with open(f"../input/{out}/contacts_{name}.json", "w") as f:
        f.write(json.dumps(d))

def create_new_contact_mtx(out):
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
    write_mtx(new_home, "home", out)
    write_mtx(new_other, "other", out)

    write_mtx(new_home, "home", f"{out}/district")
    write_mtx(new_other, "other", f"{out}/district")

    write_mtx(new_home, "home", f"{out}/district_eigen")
    write_mtx(new_other, "other", f"{out}/district_eigen")

def create_config(N, K, out):
    d = {
        "commuting_file": "commuting_KSH.json",
        "populations_file": "populations_KSH.json",
        "contacts_file_home": "contacts_home.json",
        "contacts_file_other": "contacts_other.json",
        "Npop": str(N[0]),
        "K":str(K)
    }
    with open(f"../input/{out}/config.json", "w") as f:
        f.write(json.dumps(d, indent=4))
    
    d["Npop"] = str(N[1])
    with open(f"../input/{out}/district/config.json", "w") as f:
        f.write(json.dumps(d, indent=4))
    with open(f"../input/{out}/district_eigen/config.json", "w") as f:
        f.write(json.dumps(d, indent=4))
# === MAIN ===
th = 10000
dest_folder = f"hun_{th}"
if(not os.path.exists(f"../input/{dest_folder}/district")):
    os.makedirs(f"../input/{dest_folder}/district")
if(not os.path.exists(f"../input/{dest_folder}/district_eigen")):
    os.makedirs(f"../input/{dest_folder}/district_eigen")

np.random.seed(1)
# Creates nodes with SEIR states, with given infection distribution
population, place_id_dict, big_cities, district_pops, district_id_dict = create_population_dict(
    sett_types, population_th=th,
    num_I = 10000,
    num_L=int(10000*(4/2.5)),
    Ks = [4,5],
    budapest = False,
    out=dest_folder)

cities = set(place_id_dict.keys())
print(f"Number of cities: {len(cities)}")
print(f'Number of age groups {len(population["populations"][1]["age"])}')

create_commuting(cities, place_id_dict, big_cities, district_pops, district_id_dict, out=dest_folder)
create_new_contact_mtx(out=dest_folder)
create_config(N=(len(cities), len(district_id_dict)), K=8, out=dest_folder)
