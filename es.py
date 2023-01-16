"""
Evolution Strategies
"""
import numpy as np


class ES:

    def __init__(self,
                 func,
                 dim,
                 seed,
                 children_recomb='D',
                 sigma_recomb='I',
                 select='NE',
                 mu=20,
                 l_mul=7,
                 e=0.01,
                 omega=0.5,
                 limit=(-500, 500),
                 evals_max=15000) -> None:

        assert children_recomb in ['D', 'GD']
        assert sigma_recomb in ['I', 'GI']
        assert select in ['NE', 'E']

        # Cost function and feasible region
        self.cost_func = func
        self.lower = limit[0]
        self.upper = limit[1]
        self.dim = dim

        # Optimisation parameters
        self.mu = mu
        self.l = l_mul * self.mu
        self.chi0 = np.random.normal()
        self.chis = np.random.normal(size=self.dim)
        self.omega = omega
        self.tau = 1 / np.sqrt(2 * np.sqrt(self.dim))
        self.tau_prime = 1 / np.sqrt(2 * self.dim)
        self.beta = 0.0873
        self.e = e

        # Population recombination
        if children_recomb == 'D':
            self.children_recomb = self.discrete
        else:
            self.children_recomb = self.global_discrete

        # Strategy parameters recombination
        if sigma_recomb == 'I':
            self.sigma_recomb = self.intermediate
        else:
            self.sigma_recomb = self.global_intermediate

        # Selection of new parents
        if select == 'NE':
            self.select = self.select_nonelitist
        else:
            self.select = self.select_elitist

        # Counts
        self.evals = 0
        self.evals_max = evals_max
        self.population = 1
        self.converge = False

        # Initialise parent population and sort by energy
        parents = []
        seeds_es = np.loadtxt("seeds_es.txt", dtype=int)
        reverse = 1
        for i in range(mu):
            seeds_es += np.random.randint(seed)
            np.random.seed(seeds_es[i])
            parent = self.generate_feasible()
            while i != 0 and np.any(np.all(parent == parents, axis=1)):
                # Parent already in parents list, ignoring first parent
                np.random.seed(seeds_es[-reverse])
                reverse += 1
                parent = self.generate_feasible()
            parents.append(parent)

        parents = np.array(parents)
        energy = self.cost_func(parents)
        self.evals += len(parents)
        sigma = np.ones((self.mu, self.dim))

        self.parents, self.parents_energy, self.parents_sigma = self.sort_population(
            parents, energy, sigma)

        # Archives
        self.hist = []
        self.best = []

        # Update best and archive
        self.best_x = self.parents[0]
        self.best_energy = self.parents_energy[0]
        self.ave_energy = np.mean(self.parents_energy)
        self.best.append(
            (self.population, self.best_x, self.best_energy, self.ave_energy))
        self.hist.append((self.population, self.parents, self.parents_energy,
                          self.parents_sigma))

        while self.evals < self.evals_max and self.converge == False:
            # Generate children
            self.generate_children()

            # Select and sort parents
            self.select()
            self.population += 1

            # Get best parent and archive
            self.best_x = self.parents[0]
            self.best_energy = self.parents_energy[0]
            self.ave_energy = np.mean(self.parents_energy)
            self.best.append((self.population, self.best_x, self.best_energy,
                              self.ave_energy))
            self.hist.append((self.population, self.parents,
                              self.parents_energy, self.parents_sigma))

            # Check convergence
            self.check_convergence()

    def generate_feasible(self):
        return np.random.uniform(self.lower, self.upper, self.dim)

    def is_feasible(self, child):
        return np.min(child) > self.lower and np.max(child) < self.upper

    def sort_population(self, population, energy, sigma):
        sorted_ind = np.argsort(energy)
        sorted_population = population[sorted_ind]
        sorted_energy = energy[sorted_ind]
        sorted_sigma = sigma[sorted_ind]

        return sorted_population, sorted_energy, sorted_sigma

    # Recombination methods
    def discrete(self, parents):
        parent_pair = np.random.randint(self.mu, size=2)
        parent1 = parents[parent_pair[0]]
        parent2 = parents[parent_pair[1]]

        child = []
        for i in range(self.dim):
            if np.random.uniform() <= 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return np.array(child)

    def global_discrete(self, parents):
        child = []
        for i in range(self.dim):
            parent = parents[np.random.randint(self.mu)]
            child.append(parent[i])
        return np.array(child)

    def intermediate(self, parents):
        parent_pair = np.random.randint(self.mu, size=2)
        parent1 = parents[parent_pair[0]]
        parent2 = parents[parent_pair[1]]

        child = self.omega * parent1 + (1 - self.omega) * parent2
        return np.array(child)

    def global_intermediate(self, parents):
        child = []
        for i in range(self.dim):
            parent_pair = np.random.randint(self.mu, size=2)
            parent1 = parents[parent_pair[0]]
            parent2 = parents[parent_pair[1]]
            child.append(self.omega * parent1[i] +
                         (1 - self.omega) * parent2[i])
        return np.array(child)

    # Mutation
    def mutate(self, child, child_sigma):
        # Mutate strategy parameters
        child_sigma *= np.exp(self.tau_prime * self.chi0 +
                              self.tau * self.chis)

        # Mutate child
        n = np.array(
            [np.random.normal(0, child_sigma[i]) for i in range(self.dim)])
        mutate_child = child + n

        # Reject mutation if mutated_child is not in feasible region
        if not self.is_feasible(mutate_child):
            mutate_child = child

        return mutate_child, child_sigma

    # Selection methods
    def select_nonelitist(self):
        sorted_population, sorted_energy, sorted_sigma = self.sort_population(
            self.children, self.children_energy, self.children_sigma)
        self.parents = sorted_population[:self.mu]
        self.parents_energy = sorted_energy[:self.mu]
        self.parents_sigma = sorted_sigma[:self.mu]

    def select_elitist(self):
        population = np.concatenate((self.children, self.parents))
        energy = np.concatenate((self.children_energy, self.parents_energy))
        sigma = np.concatenate((self.children_sigma, self.parents_sigma))
        sorted_population, sorted_energy, sorted_sigma = self.sort_population(
            population, energy, sigma)
        self.parents = sorted_population[:self.mu]
        self.parents_energy = sorted_energy[:self.mu]
        self.parents_sigma = sorted_sigma[:self.mu]

    # Generate children population through recombination and mutation
    def generate_children(self):
        self.children = []
        self.children_sigma = []

        for i in range(self.l):
            # Recombination
            child = self.children_recomb(self.parents)
            child_sigma = self.sigma_recomb(self.parents_sigma)

            mutate_child, mutate_sigma = self.mutate(child, child_sigma)

            self.children.append(mutate_child)
            self.children_sigma.append(mutate_sigma)

        self.children = np.array(self.children)
        self.children_energy = np.array(self.cost_func(self.children))
        self.children_sigma = np.array(self.children_sigma)

        self.evals += len(self.children)

    # Check for convergence of population
    def check_convergence(self):
        if np.abs(np.max(self.parents_energy) -
                  np.min(self.parents_energy)) <= self.e:
            self.converge = True