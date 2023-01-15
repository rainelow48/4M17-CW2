import numpy as np
import pandas as pd

DEST = "C:\\D\\Cambridge\\Part IIB\\Coursework\\4M17 Practical Optimisation\\4M17-CW2\\Results-ES\\"
CHILDREN_RECOMB = ["D", "GD"]
SIGMA_RECOMB = ["I", "GI"]
SELECT = ["NE", "E"]
DIM = [2, 6]

cases = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1),
         (1, 1, 0), (1, 1, 1)]

inv = "mu"

# Set up correct list to iterate through
if inv == "case":
    indep = cases
elif inv == "mu":
    indep = np.arange(5, 101, 5)
elif inv == "l_mul":
    indep = np.arange(5, 11, 1)
else:
    print("Error with investigating parameter")

runs = []
for ind in indep:
    if inv == "case":
        case = ind
        mu = 20
        l_mul = 7
    elif inv == "mu":
        case = cases[0]
        mu = ind
        l_mul = 7
    elif inv == "l_mul":
        case = cases[0]
        mu = 20
        l_mul = ind
    else:
        pass

    children_recomb = CHILDREN_RECOMB[case[0]]
    sigma_recomb = SIGMA_RECOMB[case[1]]
    select = SELECT[case[2]]
    dim = DIM[1]

    filename = " ".join(
        [children_recomb, sigma_recomb, select,
         str(mu),
         str(l_mul),
         str(dim)])

    df_runs = pd.read_csv(DEST + filename + " runs", index_col=0)
    runs.append(df_runs)

analysis = []
for i, df_runs in enumerate(runs):
    p = df_runs['population']
    be = df_runs['best_energy']
    rt = df_runs['running_time']
    p_ave = pd.DataFrame.mean(p)
    p_std = pd.DataFrame.std(p)
    be_ave = pd.DataFrame.mean(be)
    be_std = pd.DataFrame.std(be)
    rt_ave = pd.DataFrame.mean(rt)
    rt_std = pd.DataFrame.std(rt)

    x_best, energy_best = pd.DataFrame.min(df_runs)[1:3]
    analysis.append(
        [p_ave, p_std, be_ave, be_std, rt_ave, rt_std, x_best, energy_best])

df_an = pd.DataFrame(np.array(analysis),
                     columns=[
                         "p_ave", "p_std", "be_ave", "be_std", "rt_ave",
                         "rt_std", "x_best", "energy_best"
                     ])

pd.DataFrame.to_csv(df_an, DEST + "Analysis\\" + inv)
