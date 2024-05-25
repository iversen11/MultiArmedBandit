"""
Author: Larkin Iversen

Script to simulate epsilon greedy algorithm for Multi-Armed Bandit problem
"""

import numpy as np
from matplotlib import pyplot as plt


class BanditArm:

    def __init__(self, distribution, distribution_kwargs, bandit_label = None):
        """
        Class for different bandit arms involved in the MAB problem
        :param distribution: distribution to sample value from (e.g. normal)
        """

        self.distribution = getattr(np.random, distribution)
        self.distribution_arguments = distribution_kwargs
        self.bandit_label = bandit_label

    def return_value(self):
        """
        Return a point value from the classes distribution
        :return: value sampled from distribution
        """
        temp_value = self.distribution(**self.distribution_arguments)
        return temp_value

class BanditProblem:

    def __init__(self, bandit_list):

        self.bandits = bandit_list
        self.n_bandits = len(bandit_list)
        self.successes = None
        self.attempts = None
        self.success_rates = None
        self.bandit_selections = None
        self.explore_exploit_selections = None

    def run_epsilon_greedy(self, epsilon, iterations):
        """
        Run the epsilon greedy simulation
        :param epsilon: exploitation probability
        :param iterations: number of iterations to run
        :return:
        """

        #rest successes here in case the algorithm is conducted several time
        self.success_rates = np.zeros(self.n_bandits)
        self.successes = np.zeros(self.n_bandits)
        self.attempts = np.zeros(self.n_bandits)
        self.bandit_selections = []
        self.explore_exploit_selections = []

        def select_bandit():
            """
            Utility function to abstract away the bandit selection code
            :return: selected bandit index, string for selection type (e.g. exploration)
            """
            exploration_probability = np.random.rand()
            if exploration_probability < (1-epsilon):
                selected_bandit = np.random.randint(0, n_bandits)
                explore_exploit_selection = 'explore'

            #if exploration probability is > epsilon, exploit the current best bandit
            else:
                #check if no successes have been found, if so select random bandit
                if np.max(self.success_rates) == 0:
                    selected_bandit = np.random.randint(0, n_bandits)
                    explore_exploit_selection = 'random initialization'

                else:
                    selected_bandit = np.argmax(self.success_rates)
                    explore_exploit_selection = 'exploit'

            return selected_bandit, explore_exploit_selection

        current_iteration = 0
        while current_iteration < iterations:

            #find exploration probability, if < epsilon select a random bandit
            selected_bandit, explore_exploit_selection = select_bandit()
            outcome = self.bandits[selected_bandit].return_value()

            self.attempts[selected_bandit] += 1

            if outcome == 1:
                self.successes[selected_bandit] += 1

            #where condition is needed to avoid nans
            self.success_rates = np.divide(self.successes, self.attempts, where = self.attempts != 0)
            self.bandit_selections.append(selected_bandit)
            self.explore_exploit_selections.append(explore_exploit_selection)
            current_iteration += 1



if __name__ == '__main__':

    #modeling with Bernoulli (binomial with n = 1)
    n_bandits = 5
    bandit_list  = [BanditArm("binomial", {'n': 1, 'p': x*.1}, bandit_label = x) for x in range(n_bandits)]
    mab = BanditProblem(bandit_list)
    epsilon = 0.8
    mab.run_epsilon_greedy(epsilon, 2000)

    def chart_mab_run(bandit_run):
        """
        Plot Bandit selections over time
        :param self:
        :return: Plotted Pyplot
        """

        bandit_values = [bandit for bandit in bandit_run.bandit_selections]

        plt.figure()
        plt.xlabel('Iteration')
        plt.ylabel('Cumulative Bandit Selections Over Time')
        plt.title(f'Bandit Selections Over Time at Epsilon = {epsilon}')
        for bandit in range(len(bandit_run.bandits)):
            temp_values = [1 if x == bandit else 0 for x in bandit_values]
            temp_cumulative = np.cumsum(temp_values)
            plt.plot(temp_cumulative, label = f'Success Probability: {bandit*10} %')
        plt.legend()
        plt.show()

    chart_mab_run(mab)
