import pickle
import numpy as np
import matplotlib.pyplot as plt
from learning_maze import LearningMazeDomain
import random

SAMPLES = [5000]
DIMENSION = [30]
DISCOUNT = [0.9]
GRID_SIZES = range(10, 11)


def main():
    for discount in DISCOUNT:
        for dimension in DIMENSION:
            for grid_size in GRID_SIZES:
                for num_samples in SAMPLES:
                    # print('>>>>>>>>>>>>>>>>>>>>>>>>>> Simulation grid of size : ' +
                    #     str(grid_size) + 'x'+str(grid_size))
                    # print(
                    #     '>>>>>>>>>>>>>>>>>>>>>>>>>> dimension basis function : ' + str(dimension))
                    # print('>>>>>>>>>>>>>>>>>>>>>>>>>> discount factor : ' + str(discount))
                    height = width = grid_size
                    num_states = grid_size*grid_size
                    reward_location = 65
                    obstacles_location = [12, 15, 16, 17, 27, 37, 30, 42,
                                          43, 44, 45, 57, 58, 61, 68, 71, 72, 76, 84, 85, 88, 91]
                    walls_location = []
                    maze = LearningMazeDomain(height, width, reward_location, walls_location, obstacles_location,
                                              num_sample=num_samples)

                    all_results = {}
                    num_iterations = 4
                    for k in range(num_iterations):
                        num_steps, learned_policy, samples, distances = maze.learn_proto_values_basis(num_basis=dimension, explore=0.1,
                                                                                                      discount=discount, max_steps=500,
                                                                                                      max_iterations=100)

                        #num_steps, learned_policy, samples, distances = maze.learn_node2vec_basis(dimension=dimension)
                        all_steps_to_goal, all_samples, all_cumulative_rewards = simulate(num_states, reward_location,
                                                                                          obstacles_location, maze, learned_policy)
                        all_results[k] = {'steps_to_goal': all_steps_to_goal, 'samples': all_samples,
                                          'cumul_rewards': all_cumulative_rewards, 'learning_distances': distances}

                    display_results(all_results, grid_size,
                                    reward_location, dimension, discount, num_samples)
                    print("\n\n")
                #plot_results(pvf_all_results, grid_size, reward_location, dimension, discount, num_samples)

                # UNCOMMENT the lines below to right the results in pickle files
                # n2v_pickle = open('pickles/n2v_' + str(grid_size) + 'grid_' + str(DIMENSION) + 'dimension_' + str(DISCOUNT) + 'discount_'+ str(NUM_SAMPLE) + 'samples', 'wb')
                # pvf_pickle = open('pickles/pvf_' + str(grid_size) + 'grid_' + str(DIMENSION) + 'dimension_' + str(DISCOUNT) + 'discount_'+ str(NUM_SAMPLE) + 'samples', 'wb')
                #
                # print('Writing pickles files...')
                # pickle.dump(n2v_all_results, n2v_pickle)
                # pickle.dump(pvf_all_results, pvf_pickle)
                #
                # n2v_pickle.close()
                # pvf_pickle.close()


def display_results(all_results, grid_size, reward_location, dimension, discount, num_samples):
    mean_steps_to_goal = 0
    steps = []
    num_iterations = 4
    for i in range(num_iterations):
        step = sum((all_results[i]['steps_to_goal']).values())
        mean_steps_to_goal += step
        steps.append(step)

    mean_steps_to_goal /= (grid_size*grid_size - 23)
    mean_steps_to_goal /= num_iterations

    print("Num Samples:", num_samples)
    print("Grid Size : ", grid_size)
    print("Dimension : ", dimension)
    print("Mean steps: ", mean_steps_to_goal)
    print("Steps: ", steps)


def simulate(num_states, reward_location, obstacles_location, maze, learned_policy, max_steps=100):
    all_steps_to_goal = {}
    all_samples = {}
    all_cumulative_rewards = {}
    for state in range(num_states):
        #print("\n")
        #print(state),

        if state != reward_location and state not in obstacles_location:
            steps_to_goal = 0
            maze.domain.reset(np.array([state]))
            absorb = False
            samples = []
            states = []

            while (not absorb) and (steps_to_goal < max_steps):
                action = learned_policy.select_action(
                    maze.domain.current_state())
                sample = maze.domain.apply_action(action)
                #print(sample.next_state),

                absorb = sample.absorb
                steps_to_goal += 1
                samples.append(sample)
            all_steps_to_goal[state] = steps_to_goal
            all_samples[state] = samples
            all_cumulative_rewards[state] = np.sum([s.reward for s in samples])

    return all_steps_to_goal, all_samples, all_cumulative_rewards


def plot_results(pvf_all_results, grid_size, reward_location, dimension, discount, num_samples):
    pvf_mean_cumulative_rewards = []
    pvf_std_cumulative_rewards = []

    pvf_mean_steps_to_goal = []
    pvf_std_steps_to_goal = []

    for init_state in range(grid_size*grid_size):
        if init_state != reward_location:
            pvf_cumulative_rewards = []
            pvf_steps_to_goal = []
            for k in range(10):
                pvf_cumulative_rewards.append(
                    pvf_all_results[k]['cumul_rewards'][init_state])
                pvf_steps_to_goal.append(
                    pvf_all_results[k]['steps_to_goal'][init_state])
            pvf_mean_cumulative_rewards.append(np.mean(pvf_cumulative_rewards))
            pvf_std_cumulative_rewards.append(np.std(pvf_cumulative_rewards))
            pvf_mean_steps_to_goal.append(np.mean(pvf_steps_to_goal))
            pvf_std_steps_to_goal.append(np.std(pvf_steps_to_goal))

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)

    ax = axs[0, 0]
    ax.errorbar(sum([range(reward_location), range(grid_size, grid_size*grid_size)], []),
                pvf_mean_steps_to_goal, yerr=pvf_std_steps_to_goal, fmt='ro', ecolor='red')
    ax.set_title('pvf: number of steps')

    ax = axs[1, 0]
    ax.errorbar(sum([range(reward_location), range(grid_size, grid_size * grid_size)], []), pvf_mean_cumulative_rewards,
                yerr=pvf_std_cumulative_rewards, fmt='ro', ecolor='red')
    ax.set_title('pvf: cumulative reward')
    fig.suptitle('Grid size = ' + str(grid_size) + ', Dimension = ' +
                 str(dimension) + ', Discount =' + str(discount))
    plt.savefig('plots/'+str(grid_size) + 'grid_' + str(dimension) + 'dimension_' + str(discount) + 'discount_' + str(
        num_samples) + 'samples.pdf')


if __name__ == "__main__":
    main()
