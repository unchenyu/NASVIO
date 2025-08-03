import os
import copy
import random
import numpy as np
from tqdm import tqdm

from nas.utils.utils import write_log


# Search for model with minimum FLOPs given performance constraint
class EvolutionFinderV2:
    def __init__(self, arch_manager, efficiency_predictor, metric_predictor, log_path, **kwargs):
        self.arch_manager = arch_manager
        self.efficiency_predictor = efficiency_predictor
        self.metric_predictor = metric_predictor
        self.log_path = log_path

        # evolution hyper-parameters
        self.arch_mutate_prob = kwargs.get("arch_mutate_prob", 0.2)
        self.population_size = kwargs.get("population_size", 500)
        self.max_time_budget = kwargs.get("max_time_budget", 20)
        self.parent_ratio = kwargs.get("parent_ratio", 0.25)
        self.mutation_ratio = kwargs.get("mutation_ratio", 0.5)

    def write_log(self, log_str, prefix="evo", should_print=False, mode='a'):
        write_log(self.log_path, log_str, prefix, should_print, mode)

    def update_hyper_params(self, new_param_dict):
        self.__dict__.update(new_param_dict)

    def random_valid_sample(self, constraint):
        while True:
            sample = self.arch_manager.random_sample_arch()
            metric = self.metric_predictor.predict_metric([sample])
            if metric <= constraint:
                return sample, metric

    def mutate_sample(self, sample, constraint):
        while True:
            new_sample = copy.deepcopy(sample)

            self.arch_manager.mutate_arch(new_sample, self.arch_mutate_prob)

            metric = self.metric_predictor.predict_metric([new_sample])
            if metric <= constraint:
                return new_sample, metric

    def crossover_sample(self, sample1, sample2, constraint):
        while True:
            new_sample = copy.deepcopy(sample1)
            for key in new_sample.keys():
                if not isinstance(new_sample[key], list):
                    new_sample[key] = random.choice([sample1[key], sample2[key]])
                else:
                    for i in range(len(new_sample[key])):
                        new_sample[key][i] = random.choice(
                            [sample1[key][i], sample2[key][i]]
                        )

            metric = self.metric_predictor.predict_metric([new_sample])
            if metric <= constraint:
                return new_sample, metric

    def run_evolution_search(self, constraint, verbose=False, **kwargs):
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        self.update_hyper_params(kwargs)

        mutation_numbers = int(round(self.mutation_ratio * self.population_size))
        parents_size = int(round(self.parent_ratio * self.population_size))

        best_valids = [100]
        population = []  # (validation, sample, latency) tuples
        child_pool = []
        metric_pool = []
        best_info = None
        if verbose:
            print("Generate random population...")
        self.write_log("Generate random population...")
        for i in range(self.population_size):
            print(i)
            sample, metric = self.random_valid_sample(constraint)
            child_pool.append(sample)
            metric_pool.append(metric)

        efficiencies = []
        for child in child_pool:
            efficiencies.append(self.efficiency_predictor.get_efficiency(child))

        for i in range(self.population_size):
            population.append((efficiencies[i], child_pool[i], metric_pool[i]))

        if verbose:
            print("Start Evolution...")
        self.write_log("Start Evolution...")
        # After the population is seeded, proceed with evolving the population.
        with tqdm(
            total=self.max_time_budget,
            desc="Searching with constraint (%s)" % constraint,
            disable=(not verbose),
        ) as t:
            for i in range(self.max_time_budget):
                parents = sorted(population, key=lambda x: x[0])[:parents_size]
                efficiency = parents[0][0]
                t.set_postfix({"efficiency": parents[0][0]})
                if not verbose and (i + 1) % 100 == 0:
                    print("Iter: {} Efficiency: {}".format(i + 1, parents[0][0]))
                self.write_log("Iter: {} Efficiency: {}".format(i + 1, parents[0][0]))

                if efficiency < best_valids[-1]:
                    best_valids.append(efficiency)
                    best_info = parents[0]
                else:
                    best_valids.append(best_valids[-1])

                population = parents
                child_pool = []
                metric_pool = []

                for j in range(mutation_numbers):
                    par_sample = population[np.random.randint(parents_size)][1]
                    # Mutate
                    new_sample, metric = self.mutate_sample(par_sample, constraint)
                    child_pool.append(new_sample)
                    metric_pool.append(metric)

                for j in range(self.population_size - mutation_numbers):
                    par_sample1 = population[np.random.randint(parents_size)][1]
                    par_sample2 = population[np.random.randint(parents_size)][1]
                    # Crossover
                    new_sample, metric = self.crossover_sample(
                        par_sample1, par_sample2, constraint
                    )
                    child_pool.append(new_sample)
                    metric_pool.append(metric)

                efficiencies = []
                for child in child_pool:
                    efficiencies.append(self.efficiency_predictor.get_efficiency(child))

                for j in range(self.population_size):
                    population.append(
                        (efficiencies[j], child_pool[j], metric_pool[j])
                    )

                t.update(1)

        self.write_log("Best valid efficiency: {}".format(best_valids), should_print=True)
        self.write_log("Best info: {}".format(best_info), should_print=True)
        return best_valids, best_info