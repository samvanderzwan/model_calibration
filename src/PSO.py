import numpy as np

class PSO:
    def __init__(self, func, swarm_size, lb, ub):
        assert len(lb) == len(ub), 'Lower- and upper-bounds must be the same length'
        assert hasattr(func, '__call__'), 'Invalid function handle'
        assert np.all(ub > lb), 'All upper-bound values must be greater than lower-bound values'
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.v_high = np.abs(self.ub - self.lb)
        self.v_low = -self.v_high

        self.swarm_size = swarm_size
        self.dim = len(lb)  # the number of dimensions each particle has
        self.x = np.random.rand(self.swarm_size, self.dim)  # particle positions
        self.v = np.zeros_like(self.x)  # particle velocities
        self.best_poss = np.zeros_like(self.x)  # best particle positions
        self.fp = np.zeros(self.swarm_size)  # best particle function values
        self.best_pos = []  # best position of each particle
        self.best_func = 1e100
        self.obj = lambda x: func(x)
        self.iteration = 0

        self.omega = 0.5
        self.phip = 0.5
        self.phig = 0.5

        #creating initial swarm
        for i in range(self.swarm_size):
            # Initialize the particle's position
            self.x[i, :] = self.lb + self.x[i, :] * (self.ub - self.lb)
            # Initialize the particle's best known position
            self.best_poss[i, :] = self.x[i, :]
            self.v[i, :] = self.v_low + np.random.rand(self.dim) * (self.v_high - self.v_low)

        # Calculate the objective's value at the current particle's
        fp = self.obj(self.x)

    def update_particles_pos(self):
        rp = np.random.rand(self.swarm_size, self.dim)
        rg = np.random.rand(self.swarm_size, self.dim)
        for i in range(self.swarm_size):
            # Update the particle's velocity
            self.v[i, :] = self.omega * self.v[i, :] + self.phip * rp[i, :] * (self.p[i, :] - self.x[i, :]) + \
                           self.phig * rg[i, :] * (self.best_pos - self.x[i, :])
            self.x[i, :] = self.x[i, :] + self.v[i, :]
            # ensuring the particles stay within bounds
            mark1 = self.x[i, :] < self.lb
            mark2 = self.x[i, :] > self.ub
            self.x[i, mark1] = self.lb[mark1]
            self.x[i, mark2] = self.ub[mark2]
        self.iteration += 1

    def calc_objective(self):
        cur_obj = self.obj(self.x)
        #check for best objective of each aprticle and total
        for i in range(self.swarm_size):
            if cur_obj[i] < self.fp[i]:
                self.fp[i] = cur_obj[i].copy()
                self.best_poss[i] = self.x[i].copy()
                if cur_obj[i] < self.best_func:
                    self.best_pos = cur_obj[i].copy()
                    self.best_pos = self.x[i].copy()

