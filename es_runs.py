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

inv = "case"

# Set up correct list to iterate through
if inv == "case":
    indep = cases
elif inv == "mu":
    indep = np.arange(10, 51, 5)
elif inv == "l_mul":
    indep = np.arange(5, 11, 1)
else:
    print("Error with investigating parameter")

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
    sf = SF()

    filename = " ".join(
        [children_recomb, sigma_recomb, select,
         str(mu), str(l_mul)])
    print("Running " + filename)

    population = []
    best_x = []
    best_energy = []
    running_times = []
    for i in range(50):
        start = datetime.now()
        es = ES(func=sf.cost_es,
                dim=dim,
                children_recomb=children_recomb,
                sigma_recomb=sigma_recomb,
                select=select,
                mu=mu,
                l_mul=l_mul)
        end = datetime.now()
        running_time = timedelta.total_seconds(end - start)

        p, bx, be = es.best[-1]
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
    df_best = pd.DataFrame(es.best,
                           columns=["population", "best_x", "best_energy"])
    df_hist = pd.DataFrame(
        es.hist,
        columns=["population", "parents", "parents_energy", "parents_sigma"])

    pd.DataFrame.to_csv(df_runs, DEST + filename + " runs")
    pd.DataFrame.to_csv(df_best, DEST + filename + " best")
    pd.DataFrame.to_csv(df_hist, DEST + filename + " hist")