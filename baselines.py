import datetime
import os
import pickle
import random
import time

import gym
import numpy as np


def get_info(env, args):
    return {
        'n_states': env.observation_space.n,
        'n_actions': env.action_space.n,
        'gamma': args.gamma,
        'action_space': env.action_space,
        'eps': args.epsilon,
    }


class RandomAgent:
    def __init__(self, info):
        self.info = info

    def act(self, state):
        return self.info['action_space'].sample()

    def update_prior(self, s, a, r):
        pass

class OptimalAgent:
    def __init__(self, info):
        pass

    def act(self, state):
        return 0

    def update_prior(self, s, a, r):
        pass

class EpsilonGreedyBandit:
    def __init__(self, info):
        self.info = info
        self.params = {}
        for s in range(info['n_states']):
            for a in range(info['n_actions']):
                self.params[(s, a)] = (10, 0)

    def act(self, state):
        if random.random() < self.info['eps']:
            return self.info['action_space'].sample()

        hat_theta = [self.params[(state, a)][0] for a in range(self.info['n_actions'])]
        return np.argmax(hat_theta)

    def update_prior(self, state, action, reward):
        mean, count = self.params[(state, action)]
        mean = (mean * count + reward) / (count + 1)
        count += 1
        self.params[action] = (mean, count)


AGENT_CLS = {
    'random': RandomAgent,
    'optimal': OptimalAgent,
    'mab': EpsilonGreedyBandit,
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', '-a', type=str, default='random',
        help='Name of the agent to use.')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='Discount factor.')
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--verbose-freq', '-v', type=int, default=50)
    args = parser.parse_args()

    name = '{}_{:%Y.%m.%d-%H.%M.%S}'.format(args.agent, datetime.datetime.now())

    env = gym.make('NChain-v0')
    info = get_info(env, args)
    
    print('Enviroment Information:\n {}'.format(info))

    agent = AGENT_CLS[args.agent](info)

    rews, acts, rsum, t = [], [], 0, 0
    state, done = env.reset(), False
    start_time = time.time()
    while not done:
        act = agent.act(state)
        next_state, rew, done, _ =  env.step(act)
        agent.update_prior(state, act, rew)

        state = next_state
        rews.append(rew)
        acts.append(act)
        rsum += rew
        t += 1

        if t % args.verbose_freq == 0:
            print('step {}:\n'
                  '  cumulative reward = {}\n'
                  '  elapsed time = {:.2f} s'.format(t, rsum, time.time() - start_time))

    logs = {
        'rewards': rews,
        'actions': acts,
        'cumulative_reward': rsum,
        'timesteps': t,
        'elapsed_time': time.time() - start_time,
        'configs': args,
    }

    os.makedirs('./logs/', exist_ok=True)
    with open('./logs/{}.pkl'.format(name), 'wb') as f:
        pickle.dump(logs, f)