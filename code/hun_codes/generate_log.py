import numpy as np
import pandas as pd

death_ratio = np.array([0.00000000e+00, 3.36964689e-06, 2.19595034e-05, 4.49107573e-05,
               1.88422215e-04, 4.99762978e-04, 1.89895681e-03, 7.40632275e-03])

def get_inf_curve(filename, death = None, K= 8):
    df = pd.read_csv(filename, sep=',')
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
    if type(death) != None:
        return np.sum(I*death, axis=1)+np.sum(I2*death, axis=1), Is.sum(axis=1), Is2.sum(axis=1)
    else:
        return Is.sum(axis=1), Is2.sum(axis=1)

th = 10000
i = 6
city = f"KSH2_{th}/base"
district = f"KSH2_{th}/district"
district_eigen = f"KSH2_{th}/district_eigen"
death, Is, Is2 = get_inf_curve(f"../output/{city}/{i}.txt", death = death_ratio*10)
death_d, Is_d, Is2_d = get_inf_curve(f"../output/{district}/{i}.txt", death = death_ratio*10)
death_d2, Is_d2, Is2_d2 = get_inf_curve(f"../output/{district_eigen}/{i}.txt", death = death_ratio*10)

print(len(death))


from torch.utils.tensorboard import SummaryWriter

class TBLogger(SummaryWriter):
    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='',
                 global_step=0, batch_size=1, world_size=1, global_step_divider=1):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        self.global_step = global_step
        self.warned_missing_grad = False
        self.batch_size = batch_size
        self.world_size = world_size
        self.dist_bs = self.batch_size * self.world_size
        self.global_step_divider = global_step_divider

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=True, double_precision=False):
        if global_step is None:
            global_step = round(self.global_step / self.global_step_divider)
        super().add_scalar(tag, scalar_value, global_step, walltime, new_style, double_precision)

logger = TBLogger(log_dir="log")

for ind,d in enumerate(death):
    logger.add_scalar("death_city", d, global_step=ind)

for ind,d in enumerate(death_d):
    logger.add_scalar("death_simple_district", d, global_step=ind)

for ind,d in enumerate(death_d2):
    logger.add_scalar("death_eigen_district", d, global_step=ind)