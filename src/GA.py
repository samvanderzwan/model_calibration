import numpy as np
import plotly.graph_objects as go


class GA:
    def __init__(self, func, pop_size, lb, ub):
        assert len(lb) == len(ub), 'Lower- and upper-bounds must be the same length'
        assert hasattr(func, '__call__'), 'Invalid function handle'
        self.pop_size = pop_size
        self.ub = np.array(ub)
        self.lb = np.array(lb)
        self.number_of_genes = len(lb)
        assert np.all(self.ub > self.lb), 'All upper-bound values must be greater than lower-bound values'
        self.obj = lambda x: func(x)
        self.max_mut = 0.05
        self.abs_mut_change = 0.25
        self.rel_mut_change = 0.25
        self.abs_mut_change_p = 0.4
        self.best_objective = 0.0
        self.best_individual = self.lb
        self.old_fitnesses = []
        self.old_population = []
        self.times_not_changed = 0
        self.population = self.create_start_population()
        self.repro_change = self.calculate_fitness()
        self.generation = 0
        self.old_values = []
        self.old_values.append(self.repro_change)

    def create_random_ind(self):
        individual = np.log10(self.lb + np.random.rand(self.number_of_genes) * (self.ub - self.lb))
        return individual

    def absolute_mutate_ind(self, individual, mut_change):
        if np.random.rand() < mut_change:
            individual = self.create_random_ind()
        return individual

    def relative_mutate_ind(self, individual):
        if np.random.rand() < self.rel_mut_change:
            individual += (np.random.rand(self.number_of_genes) - 0.5) * 2 * self.max_mut * individual
            # check if genes are still within bounds
            for i in range(self.number_of_genes):
                if 10 ** individual[i] > self.ub[i]:
                    individual[i] = np.log10(self.ub[i])
                if 10 ** individual[i] < self.lb[i]:
                    individual[i] = np.log10(self.lb[i])
        return individual

    def calculate_fitness(self):
        self.times_not_changed += 1
        fitness = self.obj([10 ** x for x in self.population])
        max_index = fitness.index(max(fitness))
        if fitness[max_index] > self.best_objective:
            self.best_objective = fitness[max_index]
            self.best_individual = self.population[max_index]
            self.times_not_changed = 0
        print('best value is ' + str(self.best_objective))
        print('Best values are ' + str(10 ** self.best_individual))
        fitness = np.append(fitness, self.best_objective)
        self.population.append(self.best_individual)
        self.old_fitnesses.append(fitness)
        self.old_population.append(self.population)
        repro_change2 = [y / sum(fitness) for y in fitness]
        repro_change = np.cumsum(repro_change2)
        return repro_change

    def create_offspring(self):
        parent = [self.get_parent(), self.get_parent()]
        child = []
        for j in range(self.number_of_genes):
            rand_val = np.random.rand()
            child.append(rand_val * self.population[int(parent[1])][j] +
                         (1.0 - rand_val) * self.population[int(parent[1])][j])
        child = np.array(child)
        # mutation
        if parent[0] == parent[1]:
            child = self.absolute_mutate_ind(child, self.abs_mut_change_p)
        else:
            child = self.absolute_mutate_ind(child, self.abs_mut_change)
        child = self.relative_mutate_ind(child)
        return child

    def get_parent(self):
        return next(x for x, val in enumerate(self.repro_change) if val > np.random.rand())

    def create_start_population(self):
        population = []
        for i in range(self.pop_size):
            population.append(self.create_random_ind())
        return population

    def create_new_generation(self):
        # create a new generation
        self.generation += 1
        population = []
        for i in range(self.pop_size):
            population.append(self.create_offspring())
        self.old_values.append(self.repro_change)
        self.population = population
        self.repro_change = self.calculate_fitness()

    def plot_current_position(self, res_folder):

        fig = go.Figure()
        all_genens = []
        all_fitnes = []
        for population, fit in zip(self.old_population, self.old_fitnesses):
            gen1 = []
            gen2 = []
            fitnes = []
            for ind, value in zip(population, fit):
                gen1.append(10 ** ind[0])
                all_genens.append(10 ** ind[0])
                #gen2.append(10 ** ind[1])
                fitnes.append(1.0 / value)
                all_fitnes.append((1.0 / value))
            fig.add_trace(go.Scatter(x=gen1, y=fitnes, mode='markers'))
            fig.update_layout(
                xaxis_title="Wave speed ",
                yaxis_title="RMSE"
                )


            #fig.add_trace(go.Scatter(x=gen1, y=gen2,  mode='markers',
            #                         marker=dict(
            #                             size=fitnes,
            #                             sizemode='area',
            #                             sizeref=2. * self.best_objective / (40. ** 2),
            #                             sizemin=4
            #                         )
            #                         ))
        #fig = px.scatter(x=gen1, y=gen2, size=fitnes)
        fig.show()

        fig.write_image(res_folder + 'overview.png')

