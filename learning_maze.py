import lspi

import numpy as np

NUM_BASIS = 10
DEGREE = 3
DISCOUNT = .9
EXPLORE = 0
NUM_SAMPLES = 1
MAX_ITERATIONS = 1000
MAX_STEPS = 100


class LearningMazeDomain():

    def __init__(self, height, width, reward_location, walls_location,
                 obstacles_location, initial_state=None, obstacles_transition_probability=.2, num_sample=NUM_SAMPLES):

        self.domain = lspi.domains.DirectedGridMazeDomain(height, width, reward_location,
                                                          walls_location, obstacles_location, initial_state, obstacles_transition_probability)

        #Make a custom domain of directed graphs

        sampling_policy = lspi.Policy(
            self.domain, lspi.basis_functions.FakeBasis(4), DISCOUNT, 1)

        self.samples = []

        for i in range(height*width):
            if i != reward_location:
                for times in range(1, 10):
                    self.domain.reset(np.array([i]))
                    action = sampling_policy.select_action(
                        self.domain.current_state())
                    self.samples.append(self.domain.apply_action(action))

        # for i in xrange(num_sample):
        #     action = sampling_policy.select_action(self.domain.current_state())
        #     self.samples.append(self.domain.apply_action(action))

        self.random_policy_cumulative_rewards = np.sum([sample.reward for
                                                        sample in self.samples])

        self.solver = lspi.solvers.LSTDQSolver()

    def learn_proto_values_basis(self, num_basis=NUM_BASIS, discount=DISCOUNT,
                                 explore=EXPLORE, max_iterations=MAX_ITERATIONS, max_steps=NUM_SAMPLES, initial_policy=None):

        if initial_policy is None:
            initial_policy = lspi.Policy(self.domain, lspi.basis_functions.ProtoValueBasis(
                self.domain.graph, 4, num_basis), discount, explore)

        learned_policy, distances = lspi.learn(self.samples, initial_policy, self.solver,
                                               max_iterations=max_iterations)

        self.domain.reset()

        steps_to_goal = 0
        absorb = False
        samples = []
        states = []
        while (not absorb) and (steps_to_goal < max_steps):
            states.append(self.domain.current_state())
            action = learned_policy.select_action(self.domain.current_state())
            sample = self.domain.apply_action(action)
            absorb = sample.absorb
            # if absorb:
            #     print('Reached the goal in %d', steps_to_goal)
            steps_to_goal += 1
            samples.append(sample)

        return steps_to_goal, learned_policy, samples, distances

    def learn_polynomial_basis(self, degree=DEGREE, discount=DISCOUNT,
                               explore=EXPLORE, max_iterations=MAX_ITERATIONS, max_steps=NUM_SAMPLES, initial_policy=None):

        if initial_policy is None:
            initial_policy = lspi.Policy(
                self.domain, lspi.basis_functions.OneDimensionalPolynomialBasis(degree, 4), discount, explore)

        learned_policy, distances = lspi.learn(self.samples, initial_policy, self.solver,
                                               max_iterations=max_iterations)

        self.domain.reset()

        steps_to_goal = 0
        absorb = False
        samples = []
        states = []
        while (not absorb) and (steps_to_goal < max_steps):
            states.append(self.domain.current_state())
            action = learned_policy.select_action(self.domain.current_state())
            sample = self.domain.apply_action(action)
            absorb = sample.absorb
            if absorb:
                print('Reached the goal in %d', steps_to_goal)
                # print("States: ",states)
                # states = []
            steps_to_goal += 1
            samples.append(sample)

        return steps_to_goal, learned_policy, samples, distances

    def learn_node2vec_basis(self, maze=None, dimension=NUM_BASIS, walk_length=30, num_walks=10, window_size=10,
                             p=1, q=1, epochs=1, discount=DISCOUNT, explore=EXPLORE, max_iterations=MAX_ITERATIONS,
                             max_steps=NUM_SAMPLES, initial_policy=None, edgelist='lspi/graph_10_maze'):

        max_steps = 0

        if initial_policy is None:
            initial_policy = lspi.Policy(self.domain, lspi.basis_functions.Node2vecBasis(
                edgelist, num_actions=4, transition_probabilities=self.domain.transition_probabilities,
                dimension=dimension, walk_length=walk_length, num_walks=num_walks, window_size=window_size,
                p=p, q=q, epochs=epochs), discount, explore)

        #Insert the neural network model
        return initial_policy.basis.model
