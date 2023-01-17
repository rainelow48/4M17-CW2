import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEST = "C:\\D\\Cambridge\\Part IIB\\Coursework\\4M17 Practical Optimisation\\4M17-CW2\\Results-ES\\"
FIG_DEST = "C:\\D\\Cambridge\\Part IIB\\Coursework\\4M17 Practical Optimisation\\4M17-CW2\\Figures\\"
FIGSIZE = (7, 5)
FIGSIZE2 = (16, 5)
CHILDREN_RECOMB = ["D", "GD"]
SIGMA_RECOMB = ["I", "GI"]
SELECT = ["NE", "E"]
DIM = [2, 6]
cases = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1),
         (1, 1, 0), (1, 1, 1)]


# Plot average minimum objective and running times for all analysis conducted
def plot_MO_RT():
    invs = ['case', 'mu']

    for inv in invs:
        df_an = pd.read_csv(DEST + "Analysis\\" + inv, index_col=0)
        be_ave = df_an['be_ave']
        be_std = df_an['be_std']
        rt_ave = df_an['rt_ave']
        rt_std = df_an['rt_std']

        if inv == 'case':
            x = np.arange(0, 8, 1)
            xlabel = "Experiment number"
            title = "(ES case)"
            color = 'b'
        elif inv == 'mu':
            x = np.arange(5, 101, 5)
            xlabel = "Parent population size $\mu$"
            title = "(ES mu)"
            color = 'b'

        # Plot average objective
        plt.rcParams['figure.figsize'] = FIGSIZE
        title_MO = "Plot of Minimum Objectives "
        title_RT = "Plot of Running Times "
        ylabel_MO = "Average minimum objective (50 runs)"
        ylabel_RT = "Average running times (50 runs)"

        plt.bar(x,
                be_ave,
                yerr=be_std,
                color=color,
                width=0.8 * (x[1] - x[0]),
                align='center',
                alpha=0.5,
                ecolor='black',
                capsize=16 * 5 / len(x))
        plt.title(title_MO + title, fontsize=14)
        plt.xlabel(xlabel, fontsize=11)
        plt.ylabel(ylabel_MO, fontsize=11)
        plt.savefig(FIG_DEST + "ES " + inv + " MO",
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
        plt.savefig(FIG_DEST + "ES " + inv + " RT",
                    transparent=True,
                    bbox_inches='tight')
        plt.clf()


# Plot average and minimum objective against generations
def plot_ave_min(params):
    df_best = pd.read_csv(DEST + params + " best", index_col=0)
    x = df_best['population']
    be = df_best['best_energy']
    e_ave = df_best['ave_energy']

    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    ax2 = ax1.twinx()
    ax1.plot(x, e_ave, 'b')
    ax2.plot(x, be, 'r')
    ax1.set_ylabel("Average Objective", color='b', fontsize=11)
    ax2.set_ylabel("Minimum Objective", color='r', fontsize=11)
    ax1.set_xlabel('Generations', fontsize=11)
    plt.title("Average/Minimum Objective against Generations (" + params + ")",
              fontsize=14)
    plt.savefig(FIG_DEST + "ES " + params[:-2],
                transparent=True,
                bbox_inches='tight')
    plt.clf()


# 2D-SF
def func(xx, yy):
    return -xx * np.sin(np.sqrt(np.abs(xx))) - yy * np.sin(np.sqrt(np.abs(yy)))


# Plot population evolution for 2D-SF
def plot_gens(params, mu):
    df_parents = pd.read_csv(DEST + params + " parents", index_col=0)
    parents = np.array(df_parents)
    generations = len(parents)
    mid = 2
    end = 4
    start_gen = parents[0]
    mid_gen = parents[mid]
    final_gen = parents[end]

    fig, (ax1, ax2, ax3) = plt.subplots(1,
                                        3,
                                        sharex=True,
                                        sharey=True,
                                        figsize=FIGSIZE2)

    x = np.linspace(-500, 500, 1001)
    y = np.linspace(-500, 500, 1001)
    xx, yy = np.meshgrid(x, y)
    zz = func(xx, yy)

    pcm = ax1.contourf(xx, yy, zz)
    ax1.plot(start_gen[:mu], start_gen[mu:], 'or')
    ax1.set_title("Generation 1")
    ax2.contourf(xx, yy, zz)
    ax2.plot(mid_gen[:mu], mid_gen[mu:], 'or')
    ax2.set_title("Generation " + str(mid))
    ax3.contourf(xx, yy, zz)
    ax3.plot(final_gen[:mu], final_gen[mu:], 'or')
    ax3.set_title("Generation " + str(end))

    ax1.set_xlabel("x2", fontsize=11)
    ax2.set_xlabel("x2", fontsize=11)
    ax3.set_xlabel("x2", fontsize=11)
    ax1.set_ylabel("x1", fontsize=11)

    ax4 = fig.add_axes([0.93, 0.1, 0.01, 0.8])
    fig.colorbar(pcm, cax=ax4)
    fig.suptitle("Population Evolution with Generations (" + params[:-2] + ")",
                 fontsize=14)

    fig.savefig(FIG_DEST + "ES path " + params,
                transparent=True,
                bbox_inches='tight')
    fig.clf()


def main():
    # 2D-SF
    for case in cases:
        children_recomb = CHILDREN_RECOMB[case[0]]
        sigma_recomb = SIGMA_RECOMB[case[1]]
        select = SELECT[case[2]]
        dim = DIM[0]
        mu = 20
        l_mul = 7
        filename = " ".join([
            children_recomb, sigma_recomb, select,
            str(mu),
            str(l_mul),
            str(dim)
        ])
        plot_ave_min(filename)
        plot_gens(filename, mu)

    plt.clf()
    plt.close('all')

    # 6D-SF
    plot_MO_RT()  # Plot minimum objective and running times
    plt.close('all')

    # Plot average and minimum objective against generations
    for case in cases:
        children_recomb = CHILDREN_RECOMB[case[0]]
        sigma_recomb = SIGMA_RECOMB[case[1]]
        select = SELECT[case[2]]
        dim = DIM[1]
        mu = 20
        l_mul = 7
        filename = " ".join([
            children_recomb, sigma_recomb, select,
            str(mu),
            str(l_mul),
            str(dim)
        ])
        plot_ave_min(filename)


if __name__ == "__main__":
    main()