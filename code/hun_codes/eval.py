import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

K=8
death_ratio = np.array([0.00000000e+00, 3.36964689e-06, 2.19595034e-05, 4.49107573e-05,
               1.88422215e-04, 4.99762978e-04, 1.89895681e-03, 7.40632275e-03])
all_death = pd.read_csv("hun_codes/data/orszagos_halott.csv", sep=',')[:755]
death_orig = np.array(all_death[181:181+180]["Hétnapos mozgóátlag"])

def get_inf_curve(filename, death = None, K= 8):
    df = pd.read_csv(filename, sep=',')
    inf_cols = [c for c in df.columns if c[0:2]=='I_']
    Is = df.filter(inf_cols, axis=1)
    
    I = np.zeros((len(Is), K, len(Is.columns)//K))
    for c in Is.columns:
        _,city,age = c.split("_")
        I[:,int(age), int(city)] = Is.loc[:, c]
    
    I = np.sum(I, axis=2)
    print(I)
    if type(death) != None:
        return np.sum(I*death, axis=1), Is.sum(axis=1)
    else:
        return Is.sum(axis=1)

def fit(x,y):
    min_ind = 0
    MIN = 1e12
    for shift in range(25):
        orig_shifted = x[shift:len(y)+25-(25-shift)]
        l2 = np.sqrt(np.sum((orig_shifted-y)**2))
        if l2 < MIN:
            MIN = l2
            min_ind = shift
    return min_ind, MIN

fig, axs = plt.subplots(1,1,figsize=(15,10))
ax = axs

death, Is = get_inf_curve(f"output/second_wave/base", death = death_ratio*10)
print(death)
print(Is)
x = np.array(list(range(len(death))))
ax.plot(x, death, label=f"base")

shift, l2 = fit(death_orig, death)
print(f"base ==> Optimal shift: {shift} (l2 = {l2})")

lns1 = ax.plot(x, death_orig[:len(x)], label="Deaths", linewidth=4)
plt.legend()
plt.show()