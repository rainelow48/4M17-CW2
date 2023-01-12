from sf import SF
from sa import SA
import numpy as np
import matplotlib.pyplot as plt

test = np.array([[420.9687, 420.9687], [0, 0]])
sf = SF()
print(sf(test))

dim = 2
seeds = np.loadtxt("seeds.txt", dtype=int)

best_x = []
best_energy = []
for seed in seeds:
    x0 = sf.generate_feasible(dim=dim, seed=seed)
    sa = SA(x0=x0,
            func=sf.cost,
            dim=dim,
            step=50,
            gen="D",
            t_mode="KP",
            cooling="ACS")
    best_x.append(sa.best_x)
    best_energy.append(sa.best_energy)

print(best_x, best_energy)
plt.plot(best_energy)
plt.show()

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