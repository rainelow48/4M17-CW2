import numpy as np
import pandas as pd
from sf import SF
from sa import SA
from datetime import datetime, timedelta

DEST = "C:\\D\\Cambridge\\Part IIB\\Coursework\\4M17 Practical Optimisation\\4M17-CW2\\Results\\"
GEN = ["C", "D"]
T_MODE = ["W", "KP"]
COOLING = ["ECS", "ACS"]
DIM = [2, 6]

cases = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1),
         (1, 1, 0), (1, 1, 1)]

inv = "step"

# Set up correct list to iterate through
if inv == "case":
    indep = cases
elif inv == "step":
    indep = np.arange(55, 101, 5)
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
    sf = SF()

    if inv == "alpha1" or inv == "alpha2":
        filename = " ".join([
            gen, t_mode, cooling,
            str(step),
            str(dim),
            str(np.round(alpha, 3))
        ])
    else:
        filename = " ".join([gen, t_mode, cooling, str(step), str(dim)])

    print("Running " + filename)

    t_init = []
    best_x = []
    best_energy = []
    running_times = []
    seeds = np.loadtxt("seeds.txt", dtype=int)
    for seed in seeds:
        np.random.seed(seed)
        x0 = sf.generate_feasible(dim=dim)
        start = datetime.now()
        sa = SA(x0=x0,
                func=sf.cost,
                dim=dim,
                step=step,
                gen=gen,
                t_mode=t_mode,
                cooling=cooling,
                alpha=alpha)
        end = datetime.now()
        running_time = timedelta.total_seconds(end - start)

        best_x.append(sa.best_x)
        best_energy.append(sa.best_energy)
        t_init.append(sa.t_init)
        running_times.append(running_time)

    df_best = pd.DataFrame({
        "best_x": best_x,
        "best_energy": best_energy,
        "t_init": t_init,
        "running_time": running_times
    })
    df_hist = pd.DataFrame(sa.hist_all, columns=["x", "energy", "accept", "t"])

    pd.DataFrame.to_csv(df_best, DEST + filename + " best")
    pd.DataFrame.to_csv(df_hist, DEST + filename + " hist")
