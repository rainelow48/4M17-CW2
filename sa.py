"""
Simulated annealing
"""
import numpy as np


class SA:

    def __init__(self,
                 x0,
                 func,
                 dim,
                 step,
                 gen="C",
                 t_mode="KP",
                 cooling="ECS",
                 limit=(-500, 500),
                 alpha=0.9,
                 omega=2.1,
                 evals_max=15000,
                 evals_survey=200):

        assert gen in ["C", "D"]
        assert t_mode in ["KP", "W"]
        assert cooling in ["ECS", "ACS"]

        # Cost function and feasible region
        self.cost_func = func
        self.lower = limit[0]
        self.upper = limit[1]
        self.dim = dim

        # Optimisation parameters
        self.gen = gen  # Solution generation
        self.t_mode = t_mode  # Initial temperature
        self.cooling = cooling  # Cooling schedule
        self.Lk_max = 1000  # Markov chain length
        self.accept_min = 0.6 * self.Lk_max  # Minimum acceptance per markov chain
        # Solution generation parameters
        self.step = step
        self.alpha = alpha
        self.omega = omega
        self.D = np.ones(self.dim) * self.step
        self.R = np.zeros(self.dim)

        # Counts
        self.evals = 0  # Objective function evaluations
        self.evals_max = evals_max
        self.evals_survey = evals_survey
        self.accept = 0  # Counts of accepted solutions
        self.Lk = 0
        self.new_best = False

        # Initialisation
        self.current_x = x0
        self.current_energy = self.cost_func(self.current_x)
        self.best_x = self.current_x
        self.best_energy = self.current_energy

        # Archives
        self.survey_energies = []
        self.hist_t = []
        self.hist_energy = []
        self.hist_all = []
        self.hist_accepted = []

        # Solution generation
        if self.gen == "C":
            self.move = self.move_c
            self.accept_move = self.accept_c
        else:
            self.move = self.move_d
            self.accept_move = self.accept_d

        # Setting initial temperature by surveying
        for i in range(self.evals_survey):
            new_x = self.move()
            new_energy = self.cost_func(new_x)
            self.survey_energies.append(new_energy)

        if self.t_mode == "KP":  # Kirkpatrick
            e = np.array(self.survey_energies)
            de = e[1:] - e[:-1]
            self.t = -np.mean(de[de > 0]) / np.log(0.8)
        else:  # White
            self.t = np.std(self.survey_energies)

        # Cooling schedule
        if self.cooling == "ECS":  # Exponential cooling schedule
            self.update_t = self.ecs
        else:  # Adaptive cooling schedule
            self.update_t = self.acs

        # Add initial point
        self.hist_accepted.append(
            (self.current_x, self.current_energy, True, self.t))
        self.hist_all.append(
            (self.current_x, self.current_energy, True, self.t))
        self.hist_t.append(self.t)
        self.hist_energy.append(self.current_energy)

        while self.evals < self.evals_max:  # and solution is not improving
            # # Termination criterion:
            # # - No improvement within an entire markov chain at one temperature and
            # # - Solution acceptance ratio falling below threshold 1e-3
            # if self.Lk == self.Lk_max and self.new_best == False and self.accept / self.Lk <= 1e-3:
            #     print("no improvement")
            #     break

            # Update temperature based on cooling schedule and reset counts
            if self.Lk >= self.Lk_max or self.accept >= self.accept_min:
                self.update_t()
                self.Lk = 0
                self.accept = 0
                self.new_best = False

            # Get new solution and energy
            new_x = self.move()
            new_energy = self.cost_func(new_x)
            dE = new_energy - self.current_energy

            # Increment counts
            self.Lk += 1
            self.evals += 1

            # Check if accept, update where necessary
            accept = self.accept_move(dE)
            if accept:
                self.current_x = new_x
                self.current_energy = new_energy
                self.accept += 1

                # Update archives
                self.hist_accepted.append(
                    (self.current_x, self.current_energy, accept, self.t))
                self.hist_t.append(self.t)
                self.hist_energy.append(self.current_energy)
            self.hist_all.append((new_x, new_energy, accept, self.t))

            # Check if new solution is the best solution, update where necessary
            if self.current_energy < self.best_energy:
                self.best_energy = self.current_energy
                self.best_x = self.current_x
                self.new_best = True

    # Solution generation 1: x_(i+1) = x_i + Cu
    def move_c(self):
        u = np.random.uniform(-1, 1, self.dim)
        new_x = self.current_x + self.step * u

        # Check if new solution is within feasible region
        if np.amax(new_x) <= self.upper and np.amin(new_x) >= self.lower:
            return new_x
        else:
            return self.move_c()

    def accept_c(self, dE):
        p = np.exp(-dE / self.t)
        return np.random.uniform() <= p

    # Solution generation 2: x_(i+1) = x_i + Du
    def move_d(self):
        u = np.random.uniform(-1, 1, self.dim)
        new_x = self.current_x + self.D * u

        # Check if new solution is feasible and update R
        if np.amax(new_x) <= self.upper and np.amin(new_x) >= self.lower:
            self.R = np.abs(self.D * u)
            return new_x
        else:
            return self.move_d()

    def accept_d(self, dE):
        d_bar = np.linalg.norm(self.R)
        p = np.exp(-dE / (self.t * d_bar))

        accept = np.random.uniform() <= p
        # Update D if accepting new solution
        if accept:
            self.D = (1 -
                      self.alpha) * self.D + self.alpha * self.omega * self.R
        return accept

    # Cooling schemes:
    def ecs(self):  # Exponential cooling scheme
        self.t *= self.alpha

    def acs(self):  # Adaptive cooling scheme
        sigma = np.std(self.hist_energy[-self.Lk:])
        self.t *= max(0.5, np.exp(-0.7 * self.t / sigma))
