import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEST = "C:\\D\\Cambridge\\Part IIB\\Coursework\\4M17 Practical Optimisation\\4M17-CW2\\Results\\"
FIG_DEST = "C:\\D\\Cambridge\\Part IIB\\Coursework\\4M17 Practical Optimisation\\4M17-CW2\\Figures\\"
FIGSIZE = (7, 5)
FIGSIZE2 = (6.5, 5)
GEN = ["C", "D"]
T_MODE = ["W", "KP"]
COOLING = ["ECS", "ACS"]
DIM = [2, 6]
cases = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1),
         (1, 1, 0), (1, 1, 1)]


def plot_MO_RT():
    invs = ['case', 'step', 'alpha1', 'alpha2']

    for inv in invs:
        df_an = pd.read_csv(DEST + "Analysis\\" + inv, index_col=0)
        e_ave = df_an["energy_ave"]
        e_std = df_an['energy_std']
        rt_ave = df_an['rt_ave']
        rt_std = df_an['rt_std']

        if inv == 'case':
            x = np.arange(0, 8, 1)
            xlabel = "Experiment number"
            title = "(SA case)"
            color = 'b'

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
        plt.rcParams['figure.figsize'] = FIGSIZE
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


def plot_obj_temp(params):
    df_hist = pd.read_csv(DEST + params + " hist", index_col=0)
    accepted = df_hist[df_hist['accept'] == True]
    obj = accepted['energy']
    t = accepted['t']

    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    ax2 = ax1.twinx()
    ax1.plot(obj, 'b')
    ax2.plot(t, 'r')
    ax1.set_ylabel("Objective", color='b', fontsize=11)
    ax2.set_ylabel("Temperature", color='r', fontsize=11)
    ax1.set_xlabel('Iterations', fontsize=11)
    plt.title("Objective/Temperature against Iterations (" + params[:-2] + ")",
              fontsize=14)
    plt.savefig(FIG_DEST + "SA " + params,
                transparent=True,
                bbox_inches='tight')
    plt.clf()


def func(xx, yy):
    return -xx * np.sin(np.sqrt(np.abs(xx))) - yy * np.sin(np.sqrt(np.abs(yy)))


def plot_path(params):
    df_hist = pd.read_csv(DEST + params + " hist", index_col=0)
    accepted = df_hist[df_hist['accept'] == True]
    step = 50
    x1 = np.array(accepted['x1'])
    x2 = np.array(accepted['x2'])

    x = np.linspace(-500, 500, 1001)
    y = np.linspace(-500, 500, 1001)
    xx, yy = np.meshgrid(x, y)
    zz = func(xx, yy)

    plt.rcParams['figure.figsize'] = FIGSIZE2
    plt.contourf(xx, yy, zz)
    plt.colorbar()
    plt.plot(x1[::step], x2[::step], '--xk', label="Best solution")
    plt.plot(x1[::step][0], x2[::step][0], '^', color='orange', label="Start")
    plt.plot(x1[-1], x2[-1], 'or', label="End")
    plt.legend(ncol=3, loc="lower center")
    plt.xlabel("x2", fontsize=11)
    plt.ylabel("x1", fontsize=11)
    plt.title("Path taken (" + params[:-2] + ")", fontsize=14)
    plt.savefig(FIG_DEST + "SA path " + params,
                transparent=True,
                bbox_inches='tight')
    plt.clf()


def main():
    # 2D-SF
    for case in cases:
        gen = GEN[case[0]]
        t_mode = T_MODE[case[1]]
        cooling = COOLING[case[2]]
        step = 50
        dim = DIM[0]
        filename = " ".join([gen, t_mode, cooling, str(step), str(dim)])
        plot_obj_temp(filename)
        plot_path(filename)

    # 6D-SF
    plot_MO_RT()  # Plot minimum objective and running times

    # Plot objective and temperature against iterations
    for case in cases:
        gen = GEN[case[0]]
        t_mode = T_MODE[case[1]]
        cooling = COOLING[case[2]]
        step = 50
        dim = DIM[1]
        filename = " ".join([gen, t_mode, cooling, str(step), str(dim)])
        plot_obj_temp(filename)


if __name__ == "__main__":
    main()