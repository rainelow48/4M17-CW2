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
# for case in cases:
# for step in np.arange(5, 51, 5):
for alpha in np.arange(0.8, 1.01, 0.01):
    case = cases[0]
    gen = GEN[case[0]]
    t_mode = T_MODE[case[1]]
    cooling = COOLING[case[2]]
    dim = DIM[1]
    step = 50
    alpha = alpha
    # filename = " ".join([gen, t_mode, cooling, str(step), str(dim)])
    filename = " ".join(
        [gen, t_mode, cooling,
         str(step),
         str(dim),
         str(np.round(alpha, 3))])

    df_best = pd.read_csv(DEST + filename + " best", index_col=0)
    df_hist = pd.read_csv(DEST + filename + " hist", index_col=0)

    best.append(df_best)
    hist.append(df_hist)

for i, df_case_best in enumerate(best):
    be = df_case_best['best_energy']
    energy_ave = pd.DataFrame.mean(be)
    energy_std = pd.DataFrame.std(be)
    x_best, energy_best = pd.DataFrame.min(df_case_best)
    print(i, energy_ave, energy_std, x_best, energy_best)