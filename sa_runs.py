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

inv = "case"

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

# Run through list with varying independent variable
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

    # Set up correct file name
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

    # Run though 50 iterations with randomised initial points
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

        # Store relevant data
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

    # Split solution into 2 columns for 2D-SF to facilitate plotting
    if dim == 2:
        x, energy, accept, t = zip(*sa.hist_all)
        xs = np.array(x)
        df_hist = pd.DataFrame({
            'x1': xs[:, 0],
            'x2': xs[:, 1],
            'energy': energy,
            'accept': accept,
            't': t
        })
    else:
        df_hist = pd.DataFrame(sa.hist_all,
                               columns=["x", "energy", "accept", "t"])

    # Store relevant data in a CSV file
    pd.DataFrame.to_csv(df_best, DEST + filename + " best")
    pd.DataFrame.to_csv(df_hist, DEST + filename + " hist")
