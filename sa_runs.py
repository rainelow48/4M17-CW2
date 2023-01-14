import numpy as np
import pandas as pd
from sf import SF
from sa import SA

DEST = "C:\\D\\Cambridge\\Part IIB\\Coursework\\4M17 Practical Optimisation\\4M17-CW2\\Results\\"
GEN = ["C", "D"]
T_MODE = ["W", "KP"]
COOLING = ["ECS", "ACS"]
DIM = [2, 6]

cases = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1),
         (1, 1, 0), (1, 1, 1)]

# for case in cases:
# for step in np.arange(5, 50, 5):
# for alpha in np.arange(0.1, 1.01, 0.05):
for alpha in np.arange(0.81, 1.01, 0.01):
    case = cases[0]
    gen = GEN[case[0]]
    t_mode = T_MODE[case[1]]
    cooling = COOLING[case[2]]
    dim = DIM[1]
    step = 50
    alpha = alpha
    sf = SF()
    # filename = " ".join([gen, t_mode, cooling, str(step), str(dim)])
    filename = " ".join(
        [gen, t_mode, cooling,
         str(step),
         str(dim),
         str(np.round(alpha, 3))])
    print("Running " + filename)

    best_x = []
    best_energy = []
    seeds = np.loadtxt("seeds.txt", dtype=int)
    for seed in seeds:
        np.random.seed(seed)
        x0 = sf.generate_feasible(dim=dim)
        sa = SA(x0=x0,
                func=sf.cost,
                dim=dim,
                step=step,
                gen=gen,
                t_mode=t_mode,
                cooling=cooling,
                alpha=alpha)
        best_x.append(sa.best_x)
        best_energy.append(sa.best_energy)

    df_best = pd.DataFrame({"best_x": best_x, "best_energy": best_energy})
    df_hist = pd.DataFrame(sa.hist_all, columns=["x", "energy", "accept", "t"])

    pd.DataFrame.to_csv(df_best, DEST + filename + " best")
    pd.DataFrame.to_csv(df_hist, DEST + filename + " hist")
