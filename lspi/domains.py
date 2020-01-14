# -*- coding: utf-8 -*-
"""Contains example domains that LSPI works on."""


import abc

from random import randint, random

import numpy as np

from sample import Sample
from scipy import sparse
from pygsp import graphs
import pygsp

class Domain(object):

    r"""ABC for domains.

    Minimum interface for a reinforcement learning domain.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def num_actions(self):
        """Return number of possible actions for the given domain.

        Actions are indexed from 0 to num_actions - 1.

        Returns
        -------
        int
            Number of possible actions.
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def current_state(self):
        """Return the current state of the domain.

        Returns
        -------
        numpy.array
            The current state of the environment expressed as a numpy array
            of the individual state variables.
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def apply_action(self, action):
        """Apply action and return a sample.

        Parameters
        ----------
        action: int
            The action index to apply. This should be a number in the range
            [0, num_actions())

        Returns
        -------
        sample.Sample
            Sample containing the previous state, the action applied, the
            received reward and the resulting state.
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def reset(self, initial_state=None):
        """Reset the simulator to initial conditions.

        Parameters
        ----------
        initial_state: numpy.array
            Optionally specify the state to reset to. If None then the domain
            should use its default initial set of states. The type will
            generally be a numpy.array, but a subclass may accept other types.

        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def action_name(self, action):
        """Return a string representation of the action.

        Parameters
        ----------
        action: int
            The action index to apply. This number should be in the range
            [0, num_actions())

        Returns
        -------
        str
            String representation of the action index.
        """
        pass  # pragma: no cover


class ChainDomain(Domain):

    """Chain domain from LSPI paper.

    Very simple MDP. Used to test LSPI methods and demonstrate the interface.
    The state space is a series of discrete nodes in a chain. There are two
    actions: Left and Right. These actions fail with a configurable
    probability. When the action fails to performs the opposite action. In
    other words if left is the action applied, but it fails, then the agent will
    actually move right (assuming it is not in the right most state).

    The default reward for any action in a state is 0. There are 2 special
    states that will give a +1 reward for entering. The two special states can
    be configured to appear at the end of the chain, in the middle, or
    in the middle of each half of the state space.

    Parameters
    ----------
    num_states: int
        Number of states in the chain. Must be at least 4.
        Defaults to 10 states.
    reward_location: ChainDomain.RewardLoction
        Location of the states with +1 rewards
    failure_probability: float
        The probability that the applied action will fail. Must be in range
        [0, 1]

    """

    class RewardLocation(object):

        """Location of states giving +1 reward in the chain.

        Ends:
            Rewards will be given at the ends of the chain.
        Middle:
            Rewards will be given at the middle two states of the chain.
        HalfMiddles:
            Rewards will be given at the middle two states of each half
            of the chain.

        """

        Ends, Middle, HalfMiddles = range(3)

    __action_names = ['left', 'right']

    def __init__(self, num_states=10,
                 reward_location=RewardLocation.Ends,
                 failure_probability=.1):
        """Initialize ChainDomain."""
        if num_states < 4:
            raise ValueError('num_states must be >= 4')
        if failure_probability < 0 or failure_probability > 1:
            raise ValueError('failure_probability must be in range [0, 1]')

        self.num_states = int(num_states)
        self.reward_location = reward_location
        self.failure_probability = failure_probability

        self._state = ChainDomain.__init_random_state(num_states)

    def num_actions(self):
        """Return number of actions.

        Chain domain has 2 actions.

        Returns
        -------
        int
            Number of actions

        """
        return 2

    def current_state(self):
        """Return the current state of the domain.

        Returns
        -------
        numpy.array
            The current state as a 1D numpy vector of type int.

        """
        return self._state

    def apply_action(self, action):
        """Apply the action to the chain.

        If left is applied then the occupied state index will decrease by 1.
        Unless the agent is already at 0, in which case the state will not
        change.

        If right is applied then the occupied state index will increase by 1.
        Unless the agent is already at num_states-1, in which case the state
        will not change.

        The reward function is determined by the reward location specified when
        constructing the domain.

        If failure_probability is > 0 then there is the chance for the left
        and right actions to fail. If the left action fails then the agent
        will move right. Similarly if the right action fails then the agent
        will move left.

        Parameters
        ----------
        action: int
            Action index. Must be in range [0, num_actions())

        Returns
        -------
        sample.Sample
            The sample for the applied action.

        Raises
        ------
        ValueError
            If the action index is outside of the range [0, num_actions())

        """
        if action < 0 or action >= 2:
            raise ValueError('Action index outside of bounds [0, %d)' %
                             self.num_actions())

        action_failed = False
        if random() < self.failure_probability:
            action_failed = True

        # this assumes that the state has one and only one occupied location
        if (action == 0 and not action_failed) \
                or (action == 1 and action_failed):
            new_location = max(0, self._state[0]-1)
        else:
            new_location = min(self.num_states-1, self._state[0]+1)

        next_state = np.array([new_location])

        reward = 0
        if self.reward_location == ChainDomain.RewardLocation.Ends:
            if new_location == 0 or new_location == self.num_states-1:
                reward = 1
        elif self.reward_location == ChainDomain.RewardLocation.Middle:
            if new_location == int(self.num_states/2) \
                    or new_location == int(self.num_states/2 + 1):
                reward = 1
        else:  # HalfMiddles case
            if new_location == int(self.num_states/4) \
                    or new_location == int(3*self.num_states/4):
                reward = 1

        sample = Sample(self._state.copy(), action, reward, next_state.copy())

        self._state = next_state

        return sample

    def reset(self, initial_state=None):
        """Reset the domain to initial state or specified state.

        If the state is unspecified then it will generate a random state, just
        like when constructing from scratch.

        State must be the same size as the original state. State values can be
        either 0 or 1. There must be one and only one location that contains
        a value of 1. Whatever the numpy array type used, it will be converted
        to an integer numpy array.

        Parameters
        ----------
        initial_state: numpy.array
            The state to set the simulator to. If None then set to a random
            state.

        Raises
        ------
        ValueError
            If initial state's shape does not match (num_states, ). In
            otherwords the initial state must be a 1D numpy array with the
            same length as the existing state.
        ValueError
            If part of the state has a value or 1, or there are multiple
            parts of the state with value of 1.
        ValueError
            If there are values in the state other than 0 or 1.

        """
        if initial_state is None:
            self._state = ChainDomain.__init_random_state(self.num_states)
        else:
            if initial_state.shape != (1, ):
                raise ValueError('The specified state did not match the '
                                 + 'current state size')
            state = initial_state.astype(np.int)
            if state[0] < 0 or state[0] >= self.num_states:
                raise ValueError('State value must be in range '
                                 + '[0, num_states)')
            self._state = state

    def action_name(self, action):
        """Return string representation of actions.

        0:
            left
        1:
            right

        Returns
        -------
        str
            String representation of action.
        """
        return ChainDomain.__action_names[action]

    @staticmethod
    def __init_random_state(num_states):
        """Return randomly initialized state of the specified size."""
        return np.array([randint(0, num_states-1)])


class GridMazeDomain(Domain):

    __action_names = ['right', 'up', 'left', 'down']

    def __init__(self, height, width, reward_location, walls_location, obstacles_location, initial_state=None,
                 obstacles_transition_probability=.2):

        if obstacles_transition_probability < 0 or obstacles_transition_probability > 1:
            raise ValueError('obstacles_transition_probability must be in range [0, 1]')

        self.width = width
        self.height = height
        self.num_states = int(height*width)

        self.reward_location = reward_location

        self.adjacency_list = [[] for _ in range(self.num_states)]

        self.initial_state = initial_state

        self.transition_probabilities = np.ones(self.num_states)

        self.transition_probabilities[obstacles_location] = obstacles_transition_probability

        self.transition_probabilities[walls_location] = 0.

        self.graph = graphs.Grid2d(N1=height, N2=width)

        self.weighted_graph = graphs.Grid2d(N1=height, N2=width)

        self.adjacency_matrix = np.zeros((height*width, height*width))

        with open('./lspi/graph_10_demo') as f:
            for line in f:
                array = []
                array.append([int(x) for x in line.split()])

                self.adjacency_matrix[array[0][0]][array[0][1]] = self.adjacency_matrix[array[0][1]][array[0][0]] = 1
                self.adjacency_list[array[0][0]].append(array[0][1])
                
                #If undirected
                self.adjacency_list[array[0][1]].append(array[0][0])

        for obstacle in obstacles_location:
            self.weighted_graph.W[obstacle, :] *= obstacles_transition_probability
            self.weighted_graph.W[:, obstacle] *= obstacles_transition_probability

        for wall in walls_location:
            self.weighted_graph.W[wall, :] = 0.
            self.weighted_graph.W[:, wall] = 0.

        self.weighted_graph.Ne = sparse.tril(self.weighted_graph.W).nnz

        self._state = self._init_random_state()


    def num_actions(self):
        return 4


    def current_state(self):
        return self._state


    def apply_action(self, action):
     
        if action < 0 or action >= self.num_actions():
            raise ValueError('Action index outside of bounds [0, %d)' %
                             self.num_actions())

        new_location = self.next_location(self._state[0], action)

        # in the case of failing action
        if new_location == self._state[0] or random() > self.transition_probabilities[new_location]:
            return Sample(self._state.copy(), action, 0., self._state.copy())

        next_state = np.array([new_location])

        if self.reward_location == new_location:
            reward = 100.
            absorb = True
            sample = Sample(self._state.copy(), action, reward, next_state.copy(), absorb)
            self.reset(self.initial_state)
        else:
            absorb = False
            reward = 0.
            sample = Sample(self._state.copy(), action, reward, next_state.copy(), absorb)
            self._state = next_state

        return sample

    def next_location(self, state, action):
        if action < 0 or action >= self.num_actions():
            raise ValueError('Action index outside of bounds [0, %d)' %
                             self.num_actions())

        if state < 0 or state >= self.num_states:
            raise ValueError('Action index outside of bounds [0, %d)' %
                             self.num_states)

        def check_right_end(current_state, width):
            if current_state > 0 and (current_state + 1) % width == 0:
                return True
            return False

        def check_left_end(current_state, width):
            if current_state % width == 0:
                return True
            return False

        def check_top_end(current_state, width):
            if current_state in range(width):
                return True
            return False

        def check_bottom_end(current_state, width, height):
            if current_state in range(width * (height - 1), width * (height - 1) + width):
                return True
            return False

        next_location = state
        if action == 0 and not check_right_end(state, self.width):
            next_location = state + 1

        if action == 1 and not check_top_end(state, self.width):
            next_location = state - self.width

        if action == 2 and not check_left_end(state, self.width):
            next_location = state - 1

        if action == 3 and not check_bottom_end(state, self.width, self.height):
            next_location = state + self.width

        for possible_state in self.adjacency_list[state]:
                if(next_location == possible_state):
                    return next_location

        return state

    def reset(self, initial_state=None):
   
        if initial_state is None:
            self._state = self._init_random_state()
        else:
            if initial_state.shape != (1, ):
                raise ValueError('The specified state did not match the '
                                 + 'current state size')
            state = initial_state.astype(np.int)
            if state[0] < 0 or state[0] >= self.num_states:
                raise ValueError('State value must be in range '
                                 + '[0, num_states)')
            if self.transition_probabilities[state[0]] == 0.:
                raise ValueError('Initial state cannot be an inaccessible state')
            if state[0] == self.reward_location:
                raise ValueError('Initial state cannot be an absorbing state')
            self._state = state

    def action_name(self, action):
        """Return string representation of actions.

        0:
            left
        1:
            right

        Returns
        -------
        str
            String representation of action.
        """
        return ChainDomain.__action_names[action]

    def _init_random_state(self):
        """Return randomly initialized state of the specified size."""

        random_state = randint(0, self.num_states - 1)

        while self.transition_probabilities[random_state] == 0. or self.reward_location == random_state:
            random_state = randint(0, self.num_states - 1)

        return np.array([random_state])


class DirectedGridMazeDomain(Domain):

    __action_names = ['right', 'up', 'left', 'down']

    def __init__(self, height, width, reward_location, walls_location, obstacles_location, initial_state=None,
                 obstacles_transition_probability=.2):

        if obstacles_transition_probability < 0 or obstacles_transition_probability > 1:
            raise ValueError(
                'obstacles_transition_probability must be in range [0, 1]')

        self.width = width
        self.height = height

        self.num_states = int(height*width)
        
        self.adjacency_list = [[] for _ in range(self.num_states)]
        self.reward_location = reward_location

        self.initial_state = initial_state

        self.transition_probabilities = np.ones(self.num_states)

        self.transition_probabilities[obstacles_location] = obstacles_transition_probability

        self.transition_probabilities[walls_location] = 0.

        self.adjacency_matrix = np.zeros((height*width, height*width))
        
        with open('./lspi/graph_10_demo') as f:
            for line in f:
                array = []  
                array.append([int(x) for x in line.split()])
                self.adjacency_matrix[array[0][0]][array[0][1]] = 1
                self.adjacency_matrix[array[0][1]][array[0][0]] = 1
                self.adjacency_list[array[0][0]].append(array[0][1])
                self.adjacency_list[array[0][1]].append(array[0][0])
        
        self.graph = graphs.Graph(self.adjacency_matrix)

        
        # for obstacle in obstacles_location:
        #     self.weighted_graph.W[obstacle,
        #                           :] *= obstacles_transition_probability
        #     self.weighted_graph.W[:,
        #                           obstacle] *= obstacles_transition_probability

        # for wall in walls_location:
        #     self.weighted_graph.W[wall, :] = 0.
        #     self.weighted_graph.W[:, wall] = 0.

        # self.weighted_graph.Ne = sparse.tril(self.weighted_graph.W).nnz

        self._state = self._init_random_state()

    def action_name(self, action):
        """Return string representation of actions.

        0:
            left
        1:
            right

        Returns
        -------
        str
            String representation of action.
        """
        return ChainDomain.__action_names[action]


    def num_actions(self):
        """Return number of actions.

        This domain has 2 actions.

        Returns
        -------
        int
            Number of actions

        """
        return 4

    def current_state(self):
        """Return the current state of the domain.

        Returns
        -------
        numpy.array
            The current state as a 1D numpy vector of type int.

        """
        return self._state

    def apply_action(self, action):
        """Apply the action to the grid.

        If left is applied then the occupied state index will decrease by 1.
        Unless the agent is already at 0, in which case the state will not
        change.

        If right is applied then the occupied state index will increase by 1.
        Unless the agent is already at num_states-1, in which case the state
        will not change.

        The reward function is determined by the reward location specified when
        constructing the domain.

        If failure_probability is > 0 then there is the chance for the left
        and right actions to fail. If the left action fails then the agent
        will move right. Similarly if the right action fails then the agent
        will move left.

        Parameters
        ----------
        action: int
            Action index. Must be in range [0, num_actions())

        Returns
        -------
        sample.Sample
            The sample for the applied action.

        Raises
        ------
        ValueError
            If the action index is outside of the range [0, num_actions())

        """
        if action < 0 or action >= self.num_actions():
            raise ValueError('Action index outside of bounds [0, %d)' %
                             self.num_actions())

        new_location = self.next_location(self._state[0], action)

        # in the case of failing action
        if new_location == self._state[0] or random() > self.transition_probabilities[new_location]:
            return Sample(self._state.copy(), action, 0., self._state.copy())

        next_state = np.array([new_location])

        if self.reward_location == new_location:
            reward = 100.
            absorb = True
            sample = Sample(self._state.copy(), action,
                            reward, next_state.copy(), absorb)
            self.reset(self.initial_state)

        else:
            absorb = False
            reward = 0.
            sample = Sample(self._state.copy(), action,
                            reward, next_state.copy(), absorb)
            self._state = next_state

        return sample

    def next_location(self, state, action):
        if action < 0 or action >= self.num_actions():
            raise ValueError('Action index outside of bounds [0, %d)' %
                             self.num_actions())

        if state < 0 or state >= self.num_states:
            raise ValueError('Action index outside of bounds [0, %d)' %
                             self.num_states)

        def check_right_end(current_state, width):
            if current_state > 0 and (current_state + 1) % width == 0:
                return True
            return False

        def check_left_end(current_state, width):
            if current_state % width == 0:
                return True
            return False

        def check_top_end(current_state, width):
            if current_state in range(width):
                return True
            return False

        def check_bottom_end(current_state, width, height):
            if current_state in range(width * (height - 1), width * (height - 1) + width):
                return True
            return False

        next_location = state
        if action == 0 and not check_right_end(state, self.width):
            next_location = state + 1

        if action == 1 and not check_top_end(state, self.width):
            next_location = state - self.width

        if action == 2 and not check_left_end(state, self.width):
            next_location = state - 1

        if action == 3 and not check_bottom_end(state, self.width, self.height):
            next_location = state + self.width

        
        for possible_state in self.adjacency_list[state]:
                if(next_location == possible_state):
                    return next_location
                
        return state

    def reset(self, initial_state=None):


        if initial_state is None:
            self._state = self._init_random_state()
        else:
            if initial_state.shape != (1, ):
                raise ValueError('The specified state did not match the '+'current state size')
            state = initial_state.astype(np.int)
            if state[0] < 0 or state[0] >= self.num_states:
                raise ValueError('State value must be in range '+ '[0, num_states)')
            if self.transition_probabilities[state[0]] == 0.:
                raise ValueError('Initial state cannot be an inaccessible state')
            if state[0] == self.reward_location:
                raise ValueError('Initial state cannot be an absorbing state')
            self._state = state


    def _init_random_state(self):
        """Return randomly initialized state of the specified size."""

        random_state = randint(0, self.num_states - 1)

        while self.transition_probabilities[random_state] == 0. or self.reward_location == random_state:
            random_state = randint(0, self.num_states - 1)

        return np.array([random_state])
