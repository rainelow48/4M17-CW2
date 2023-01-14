import numpy as np
import matplotlib.pyplot as plt

class Plot:
    def __init__(self) -> None:
        self.DEST = "C:\\D\\Cambridge\\Part IIB\\Coursework\\4M17 Practical Optimisation\\4M17-CW2\\Figures"
        pass
    
    def heatmap(self, func):
        x = np.linspace(-500, 500, 1001)
        y = np.linspace(-500, 500, 1001)
        xx, yy = np.meshgrid(x, y)

        zz = func(xx, yy)

        plt.contourf(xx, yy, zz)
        plt.colorbar()

    def format(self, title, xlabel, ylabel, filename, save=True):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        if save: plt.savefig(self.DEST + filename)

    def show(self):
        plt.show()