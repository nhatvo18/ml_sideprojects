"""
Created by:     Dex Vo
Date:           11/08/2018
Class:          CSC 371, Assignment 5
"""

"""
In this test, I set population size = 100, mutation probability = 0.001,
max iteration = 10000, and terminate when population's average fitness > 0.999

For the purpose of testing I find it unnecessary to wait until average fitness > 0.9999;
it will just take significantly longer.

Since each computer might have a different graphing package, in this code I just output
max_fitness and average_fitness overtime rather than graphing them. It is easy
to see that they trend upwards.

I will include a separate screenshot of the graph with the submission.
"""

import numpy as np
import random

def mutate(child, mutate_probability):
    mutation = []
    for i in child:
        if random.uniform(0, 1) < mutate_probability:
            if i == 0:
                mutation.append(1)
            else:
                mutation.append(0)
        else:
            mutation.append(i)
    return mutation

def reproduce(parent_1, parent_2, mutate_probability):
    i = random.randint(0, 5)
    j = random.randint(5, 9)
    # Two-point crossover
    child_1 = parent_1[:i] + parent_2[i:j] + parent_1[j:]
    child_2 = parent_2[:i] + parent_1[i:j] + parent_2[j:]
    # Mutate
    child_1_ = mutate(child_1, mutate_probability)
    child_2_ = mutate(child_2, mutate_probability)
    return [child_1_, child_2_]

class Population:

    def __init__(self, size):
        self.size = size
        self.population = self._build_population()

    def _build_population(self):
        p = np.random.randint(2, size = (self.size, 8))
        return p

    def calc_fitness(self):
        fitness = np.zeros(self.size)
        # convert P from binary to decimal
        for index_1, i in enumerate(self.population):
            count = self.population[index_1].size - 1
            for index_2, j in enumerate(i):
                fitness[index_1] += j * (2 ** count)
                count -= 1
        # Input n values into the function
        fitness[:] = [np.sin(np.pi * x / 256.0) for x in fitness]
        return fitness

    def normalize_fitness(self, fitness):
        # Normalize fitness
        normalized_index = np.array(fitness)
        sum = np.sum(normalized_index)
        normalized_index[:] = [x / sum for x in normalized_index]
        return normalized_index

    def find_parents(self, normalized_index):
        parents = []
        for i in range(self.size):
            sum = 0
            k = random.uniform(0, 1)
            for index, j in enumerate(normalized_index):
                sum += j
                if k <= sum:
                    parents.append(self.population[index])
                    break
        parents = np.array(parents, dtype = int)
        return parents

    def next_generation(self, parents, mutate_probability):
        next_generation = []
        male = np.array(parents[0::2])
        female = np.array(parents[1::2])
        for i in range(len(male)):
            children = reproduce(male[i].tolist(), female[i].tolist(), mutate_probability)
            next_generation += children
        next_generation = np.array(next_generation, dtype = int)
        self.population = next_generation

    def max_fitness(self, fitness):
        return np.max(fitness)

    def average_fitness(self, fitness):
        return np.average(fitness)


def evolve():
    # Parameters
    size = 100 # size must be even
    mutate_probability = 0.001
    iteration = 10000
    # Create population
    population = Population(size)
    iter = 0
    while iter < iteration:
        # Find population's fitness
        fitness = population.calc_fitness()
        max_fitness = population.max_fitness(fitness)
        aver_fitness = population.average_fitness(fitness)
        if iter % 20 == 0:
            print(max_fitness, aver_fitness, iter)
        if aver_fitness > 0.999:
            break
        # Evolve population
        normalized_fitness = population.normalize_fitness(fitness)
        parents = population.find_parents(normalized_fitness)
        population.next_generation(parents, mutate_probability)
        iter += 1

    print(max_fitness, aver_fitness, iter)


if __name__ == '__main__':
    evolve()
