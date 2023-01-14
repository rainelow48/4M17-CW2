from sf import SF
from sa import SA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# test = np.array([[420.9687, 420.9687], [0, 0], [-5, -5]])
# sf = SF()
# print(sf.cost_es(test))

# def func(xx, yy):
#     return -xx * np.sin(np.sqrt(np.abs(xx))) - yy * np.sin(np.sqrt(np.abs(yy)))

# x = np.linspace(-500, 500, 1001)
# y = np.linspace(-500, 500, 1001)
# xx, yy = np.meshgrid(x, y)

# zz = func(xx, yy)

# fig, ax1 = plt.subplots(figsize=(8, 8))
# ax2 = ax1.twinx()

# plt.contourf(xx, yy, zz)
# plt.colorbar()
# plt.show()

# dim = 2
# seeds = np.loadtxt("seeds.txt", dtype=int)
# x0 = sf.generate_feasible(dim=dim, seed=seeds[0])
# sa = SA(x0=x0,
#         func=sf.cost,
#         dim=dim,
#         step=50,
#         gen="C",
#         t_mode="W",
#         cooling="ECS")

# print(sa.best_energy, sa.best_x)
# print(np.shape(sa.hist_accepted))
# hist_accepted = pd.DataFrame(sa.hist_accepted,
#                              columns=['x', 'energy', 'accepted', 'temp'])
# hist_all = pd.DataFrame(sa.hist_all,
#                         columns=['x', 'energy', 'accepted', 'temp'])
# print(hist_accepted['x'])
# xs = hist_accepted['x'].to_numpy()

# a, b = zip(*xs)
# print(len(a), len(b))
# ax1.plot(hist_all['energy'], 'b')
# ax2.plot(hist_all['temp'], 'r')
# # plt.plot(a[::25], b[::25], '--rx', label="x")
# plt.show()
# best_x = []
# best_energy = []
# for seed in seeds:
#     np.random.seed(seed)
#     x0 = sf.generate_feasible(dim=dim)
#     sa = SA(x0=x0,
#             func=sf.cost,
#             dim=dim,
#             step=50,
#             gen="D",
#             t_mode="KP",
#             cooling="ACS")
#     best_x.append(sa.best_x)
#     best_energy.append(sa.best_energy)

# print(best_x, best_energy)
# plt.plot(best_energy)
# plt.show()

# t = sa.hist_t
# en = sa.hist_energy
# print(len(sa.hist_all), len(sa.hist_accepted))

# plt.plot(t, label="temperature")
# plt.plot(en, label="energy")
# plt.legend()
# plt.show()

# Generate unique seeds for evaluations
# seeds = []
# for i in range(50):
#     seeds.append(np.random.randint(0, 10000))

# if len(set(seeds)) == 50:
#     print("saving...")
#     np.savetxt("seeds.txt", np.array(seeds, dtype=int))