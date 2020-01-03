import pickle
import numpy as np
import matplotlib.pyplot as plt
from learning_maze import LearningMazeDomain
from DQAgent import DQAgent
import datetime
import random


num_samples = 100
DIMENSION = [10]
DISCOUNT = [0.9]
GRID_SIZES = range(10, 11)


def main():
    for discount in DISCOUNT:
        for dimension in DIMENSION:
            for grid_size in GRID_SIZES:
                print('>>>>>>>>>>>>>>>>>>>>>>>>>> Simulation grid of size : ' +
                      str(grid_size) + 'x'+str(grid_size))
                print(
                    '>>>>>>>>>>>>>>>>>>>>>>>>>> dimension basis function : ' + str(dimension))
                print('>>>>>>>>>>>>>>>>>>>>>>>>>> discount factor : ' + str(discount))
                height = width = grid_size
                num_states = grid_size*grid_size
                reward_location = 76
                obstacles_location = []
                walls_location = []
                maze = LearningMazeDomain(height, width, reward_location, walls_location, obstacles_location,
                                          num_sample=num_samples)

                all_results = {}
                num_iterations = 1
                for _ in range(num_iterations):
                    # num_steps, learned_policy, samples, distances = maze.learn_proto_values_basis(num_basis=dimension, explore=0,
                    #                                                                                                 discount=discount, max_steps=500,
                    #                                                                                                 max_iterations=200)
                    embeds = maze.learn_node2vec_basis()
                    trainDQN(maze.domain, embeds)

                    print("learnt")
                    

                display_results(all_results[num_iterations-1], grid_size,reward_location, dimension, discount, num_samples)


def simulate(num_states, reward_location, walls_location, maze, learned_policy, max_steps=100):
    all_steps_to_goal = {}
    all_samples = {}
    all_cumulative_rewards = {}
    for state in range(num_states):
        print(state)
        if state != reward_location and state not in walls_location:
            steps_to_goal = 0
            maze.domain.reset(np.array([state]))
            absorb = False
            samples = []
            while (not absorb) and (steps_to_goal < max_steps):
                action = learned_policy.select_action(
                    maze.domain.current_state())
                sample = maze.domain.apply_action(action)
                print(sample.next_state)
                absorb = sample.absorb
                steps_to_goal += 1
                samples.append(sample)
                all_steps_to_goal[state] = steps_to_goal
                all_samples[state] = samples
                all_cumulative_rewards[state] = np.sum(
                    [s.reward for s in samples])

    return all_steps_to_goal, all_samples, all_cumulative_rewards


def deepQLearning(model, env, randomMode=False, **opt):

    episodes = 100
    batch_size = 10

    start_time = datetime.datetime.now()

    for episode in range(episodes):
        print("Episode done")
        loss = 0.0
        env.reset()
        game_over = False
        # number of step for each episode
        n_step = 0
        list_action = []
        next_state = env._state
        max_iterations = 100
        while not game_over or n_step > max_iterations:
            valid_actions = env.valid_actions()
            # if not valid_actions:
            #     game_over = True
            #     print(env.map)
            #     continue

            current_state = next_state
            # Get next action
           
            if np.random.rand() < model.epsilon:
                action = random.choice(valid_actions)
            else:
                action = model.predict(current_state)
            # print("**action = {}".format(action))
            # Apply action, get reward and new envstate

            new_sample = env.apply_action(action)
            next_state = new_sample.next_state
            # print("action = {}, valid_actions = {}".format(action, valid_actions))
            if new_sample.absorb:
                game_over = True
            else:
                game_over = False

            # if DEBUG:
            #     print("--------------------------------------")
            #     print(np.reshape(current_state, newshape=(4, 4)))
            #     print("action = {},valid_action = {},reward = {}, game_over = {}".format(action, valid_actions,
            #                                                                              reward, game_over))
            #     print(np.reshape(next_state, newshape=(4, 4)))
            list_action.append(action)
            # Store episode (experience)
            model.remember(current_state, action, next_state,
                           new_sample.reward, new_sample.absorb)
            n_step += 1
            loss = model.replay(batch_size)
            # TODO: loss = model.evaluate(inputs, targets, verbose=0)
            # if e % 10 == 0:
            #     agent.save("./save/cartpole.h5")

        # if (game_status == STATE_WIN and list_action not in memory):
        #     memory.append(list_action)
        #template = "Episodes: {:03d}/{:d} |Loss: {:.4f} | Total_reward: {:3.4f} | Episodes: {:d} | Epsilon : {:.3f} | Total win: {:d} | Win rate: {:.3f} | time: {}"
        # print(template.format(episode, MAX_EPISODES - 1, loss, env.total_reward, n_step, model.epsilon,
        #                       sum(win_history),
        #                       win_rate, t))

        # Some Terminating Condition


def trainDQN(maze, embeds):
    print(embeds)
    env = maze
    if env is None:
        return
    model = DQAgent(env, embeds)
    deepQLearning(model, env)
    pass


def display_results(all_results, grid_size, reward_location, dimension, discount, num_samples):
    mean_steps_to_goal = 0

    mean_steps_to_goal = sum((all_results['steps_to_goal']).values())

    mean_steps_to_goal /= (grid_size*grid_size - 1)

    print("Grid Size : ", grid_size)
    print("Dimenstion : ", dimension)
    print("Mean steps: ", mean_steps_to_goal)


def plot_results(pvf_all_results, grid_size, reward_location, dimension, discount, num_samples):
    pvf_mean_cumulative_rewards = []
    pvf_std_cumulative_rewards = []

    pvf_mean_steps_to_goal = []
    pvf_std_steps_to_goal = []

    for init_state in range(grid_size*grid_size):
        if init_state != reward_location:
            pvf_cumulative_rewards = []
            pvf_steps_to_goal = []
            for k in range(2):
                pvf_cumulative_rewards.append(
                    pvf_all_results[k]['cumul_rewards'][init_state])
                pvf_steps_to_goal.append(
                    pvf_all_results[k]['steps_to_goal'][init_state])
            pvf_mean_cumulative_rewards.append(np.mean(pvf_cumulative_rewards))
            pvf_std_cumulative_rewards.append(np.std(pvf_cumulative_rewards))
            pvf_mean_steps_to_goal.append(np.mean(pvf_steps_to_goal))
            pvf_std_steps_to_goal.append(np.std(pvf_steps_to_goal))

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)

    ax = axs[0]
    ax.errorbar(sum([range(reward_location), range(grid_size, grid_size*grid_size)], []),
                pvf_mean_steps_to_goal, yerr=pvf_std_steps_to_goal, fmt='ro', ecolor='red')
    ax.set_title('pvf: number of steps')

    ax = axs[1]
    ax.errorbar(sum([range(reward_location), range(grid_size, grid_size * grid_size)], []), pvf_mean_cumulative_rewards,
                yerr=pvf_std_cumulative_rewards, fmt='ro', ecolor='red')
    ax.set_title('pvf: cumulative reward')
    fig.suptitle('Grid size = ' + str(grid_size) + ', Dimension = ' +
                 str(dimension) + ', Discount =' + str(discount))
    plt.savefig('plots/'+str(grid_size) + 'grid_' + str(dimension) + 'dimension_' + str(discount) + 'discount_' + str(
        num_samples) + 'samples.pdf')


if __name__ == "__main__":
    main()
