import numpy as np
import pandas as pd
from sf import SF
from es import ES
from datetime import datetime, timedelta

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
    indep = np.arange(40, 61, 5)
elif inv == "l_mul":
    indep = np.arange(5, 11, 1)
else:
    print("Error with investigating parameter")

# Run through list with varying independent variable
for ind in indep:
    if inv == "case":
        case = ind
        mu = 20
        l_mul = 7
    elif inv == "mu":
        case = cases[5]
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
    sf = SF()

    # Set up file name
    filename = " ".join(
        [children_recomb, sigma_recomb, select,
         str(mu),
         str(l_mul),
         str(dim)])
    print("Running " + filename)

    population = []
    best_x = []
    best_energy = []
    running_times = []

    # Run though 50 iterations with randomised initial population
    seeds_50 = np.loadtxt('seeds.txt', dtype=int)
    for i in range(50):
        seed = seeds_50[i]
        start = datetime.now()
        es = ES(func=sf.cost_es,
                dim=dim,
                seed=seed,
                children_recomb=children_recomb,
                sigma_recomb=sigma_recomb,
                select=select,
                mu=mu,
                l_mul=l_mul)
        end = datetime.now()
        running_time = timedelta.total_seconds(end - start)

        # Store relevant data
        p, bx, be, e_ave = es.best[-1]
        population.append(p)
        best_x.append(bx)
        best_energy.append(be)
        running_times.append(running_time)

    df_runs = pd.DataFrame({
        "population": population,
        "best_x": best_x,
        "best_energy": best_energy,
        "running_time": running_times
    })
    df_best = pd.DataFrame(
        es.best, columns=["population", "best_x", "best_energy", "ave_energy"])

    # Split solution into 2 columns for 2D-SF to facilitate plotting
    if dim == 2:
        population, parents, parents_energy, parents_sigma = zip(*es.hist)
        ps = np.array([np.array(i).T.ravel() for i in parents])
        df_parents = pd.DataFrame(ps)
        pd.DataFrame.to_csv(df_parents, DEST + filename + " parents")

    df_hist = pd.DataFrame(
        es.hist,
        columns=["population", "parents", "parents_energy", "parents_sigma"])

    # Store relevant data in a CSV file
    pd.DataFrame.to_csv(df_runs, DEST + filename + " runs")
    pd.DataFrame.to_csv(df_best, DEST + filename + " best")
    pd.DataFrame.to_csv(df_hist, DEST + filename + " hist")