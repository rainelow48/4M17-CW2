import numpy as np
import pandas as pd

DEST = "C:\\D\\Cambridge\\Part IIB\\Coursework\\4M17 Practical Optimisation\\4M17-CW2\\Results\\"
GEN = ["C", "D"]
T_MODE = ["W", "KP"]
COOLING = ["ECS", "ACS"]
DIM = [2, 6]

cases = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1),
         (1, 1, 0), (1, 1, 1)]
best = []
hist = []

invs = ['case', 'step', 'alpha1', 'alpha2']
inv = invs[0]

# Set up correct list to iterate through
if inv == "case":
    indep = cases
elif inv == "step":
    indep = np.arange(5, 101, 5)
elif inv == "alpha1":
    indep = np.arange(0.1, 1.01, 0.05)
elif inv == "alpha2":
    indep = np.arange(0.8, 1.01, 0.01)
else:
    print("Error with investigating parameter")

for ind in indep:
    if inv == "case":
        case = ind
        step = 50
        alpha = 0.9
    elif inv == "step":
        case = cases[0]
        step = ind
        alpha = 0.9
    elif inv == "alpha1" or inv == "alpha2":
        case = cases[0]
        step = 50
        alpha = ind
    else:
        pass
    gen = GEN[case[0]]
    t_mode = T_MODE[case[1]]
    cooling = COOLING[case[2]]
    dim = DIM[1]

    if inv == "alpha1" or inv == "alpha2":
        filename = " ".join([
            gen, t_mode, cooling,
            str(step),
            str(dim),
            str(np.round(alpha, 3))
        ])
    else:
        filename = " ".join([gen, t_mode, cooling, str(step), str(dim)])

    df_best = pd.read_csv(DEST + filename + " best", index_col=0)
    best.append(df_best)

    # df_hist = pd.read_csv(DEST + filename + " hist", index_col=0)
    # hist.append(df_hist)

analysis = []
for i, df_case_best in enumerate(best):
    be = df_case_best['best_energy']
    t = df_case_best['t_init']
    rt = df_case_best['running_time']
    energy_ave = pd.DataFrame.mean(be)
    energy_std = pd.DataFrame.std(be)
    t_ave = pd.DataFrame.mean(t)
    t_std = pd.DataFrame.std(t)
    rt_ave = pd.DataFrame.mean(rt)
    rt_std = pd.DataFrame.std(rt)

    x_best, energy_best, best_t_init = pd.DataFrame.min(df_case_best)[:-1]
    analysis.append([
        energy_ave, energy_std, energy_best, x_best, t_ave, t_std, rt_ave,
        rt_std
    ])
    print(i, energy_ave, energy_std, x_best, energy_best)

df_an = pd.DataFrame(np.array(analysis),
                     columns=[
                         "energy_ave", "energy_std", "energy_best", "x_best",
                         "t_ave", "t_std", "rt_ave", "rt_std"
                     ])
pd.DataFrame.to_csv(df_an, DEST + "Analysis\\" + inv)
