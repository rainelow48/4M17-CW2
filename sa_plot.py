import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEST = "C:\\D\\Cambridge\\Part IIB\\Coursework\\4M17 Practical Optimisation\\4M17-CW2\\Results\\"
FIG_DEST = "C:\\D\\Cambridge\\Part IIB\\Coursework\\4M17 Practical Optimisation\\4M17-CW2\\Figures\\"

invs = ['case', 'step', 'alpha1', 'alpha2']

for inv in invs:
    df_an = pd.read_csv(DEST + "Analysis\\" + inv, index_col=0)
    e_ave = df_an["energy_ave"]
    e_std = df_an['energy_std']
    rt_ave = df_an['rt_ave']
    rt_std = df_an['rt_std']
    figsize = (14, 5)

    if inv == 'case':
        x = np.arange(0, 8, 1)
        xlabel = "Experiment number"
        title = "(SA case)"
        color = 'b'
        figsize = (7, 5)

    elif inv == 'step':
        x = np.arange(5, 101, 5)
        xlabel = "Step"
        title = "(SA step)"
        color = 'r'

    elif inv == 'alpha1':
        x = np.arange(0.1, 1.01, 0.05)
        xlabel = "Alpha"
        title = "(SA alpha)"
        color = 'g'

    elif inv == 'alpha2':
        x = np.arange(0.8, 1.01, 0.01)
        xlabel = "Alpha"
        title = "(SA alpha)"
        color = 'g'

    # Plot average objective
    plt.rcParams['figure.figsize'] = figsize
    title_MO = "Plot of Minimum Objectives "
    title_RT = "Plot of Running Times "
    ylabel_MO = "Average minimum objective (50 runs)"
    ylabel_RT = "Average running times (50 runs)"

    plt.bar(x,
            e_ave,
            yerr=e_std,
            color=color,
            width=0.8 * (x[1] - x[0]),
            align='center',
            alpha=0.5,
            ecolor='black',
            capsize=16 * 5 / len(x))
    plt.title(title_MO + title, fontsize=14)
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel_MO, fontsize=11)
    plt.savefig(FIG_DEST + "SA " + inv + " MO",
                transparent=True,
                bbox_inches='tight')
    plt.clf()

    # Plot running times
    plt.bar(x,
            rt_ave,
            yerr=rt_std,
            width=0.8 * (x[1] - x[0]),
            color=color,
            align='center',
            alpha=0.5,
            ecolor='black',
            capsize=16 * 5 / len(x))
    plt.title(title_RT + title, fontsize=14)
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel_RT, fontsize=11)
    plt.savefig(FIG_DEST + "SA " + inv + " RT",
                transparent=True,
                bbox_inches='tight')
    plt.clf()
