from collections import defaultdict

import gym
import numpy as np


ENV = 'NChain-v0'


def encode_hyperstate(hyperstate):
    state, alphas = hyperstate
    alpha_str = '|'.join([str(a) for a in alphas])
    return '{}h{}'.format(state, alpha_str)


def decode_hyperstate(encoded):
    state, alphas = encoded.split('h')
    state = int(state)
    alphas = np.array(alphas.split('|'), dtype=np.long)
    return (state, alphas)


def get_info(env, args):
    def reward_fn(state, action, next_state):
        if action:
            return env.small
        if state < next_state:
            return 0
        return env.large

    return {
        'n_states': env.observation_space.n,
        'n_actions': env.action_space.n,
        'reward_fn': reward_fn,
        'gamma': args.gamma,
        'V_min': 0,
        'V_max': env.large * args.max_depth,
    }


def _get_index(state, action, next_state, info):
    n_states = info['n_states']
    n_actions = info['n_actions']
    return (state * n_states * n_actions +
            action * n_states +
            next_state)


def init_hyperstate(state, info, prior_alphas=None):
    n_params = info['n_states'] * info['n_actions'] * info['n_states']
    if not prior_alphas:
        prior_alphas = np.ones(n_params, dtype=np.long)
    return encode_hyperstate((state, prior_alphas))


def sample(hyperstate, action, info):
    state, alphas = decode_hyperstate(hyperstate)

    # sample next state
    start = _get_index(state, action, 0, info)
    relevant_alphas = alphas[start:start + info['n_states']]
    transition_probas = np.random.dirichlet(relevant_alphas)
    next_state = np.argmax(np.random.multinomial(1, transition_probas))
    
    # get corresponding reward
    reward = info['reward_fn'](state, action, next_state)

    return reward, state


def update_posterior(hyperstate, action, next_state, info):
    state, alphas = decode_hyperstate(hyperstate)

    idx = _get_index(state, action, next_state, info)
    new_alphas = alphas.copy()
    new_alphas[idx] += 1

    return encode_hyperstate((next_state, new_alphas))


def bellman_backup(hyperstate, params, info):
    U = np.zeros(info['n_actions'], dtype=np.float32)
    L = np.zeros(info['n_actions'], dtype=np.float32)
    for action in range(info['n_actions']):
        reward = params['R'][(hyperstate, action)]
        num_children = len(params['children'][(hyperstate, action)])
        for next_hs in params['children'][(hyperstate, action)]:
            U[action] += (reward + info['gamma'] * params['U'][next_hs]) / num_children
            L[action] += (reward + info['gamma'] * params['L'][next_hs]) / num_children
    params['U'][hyperstate] = np.max(U)
    params['L'][hyperstate] = np.min(L)
    params['U_a'][hyperstate] = np.argmax(U)
    params['L_a'][hyperstate] = np.argmax(L)
    return params


def update_params(params, hs, act, next_hs, rew, branching_factor):
    if (hs, act) not in params['count']:
        params['count'][(hs, act)] = defaultdict(int)
    params['count'][(hs, act)][next_hs] += 1

    if (hs, act) not in params['children']:
        params['children'][(hs, act)] = set()
    params['children'][(hs, act)].add(next_hs)

    if (hs, act) not in params['R']:
        params['R'][(hs, act)] = 0
    params['R'][(hs, act)] += rew / branching_factor

    return params


def fsss_rollout(hyperstate, max_depth, branching_factor, params, info):
    if max_depth == 0:
        return params
    
    if hyperstate not in params['visited']:
        params['visited'].add(hyperstate)
        for action in range(info['n_actions']):
            for _ in range(branching_factor):
                reward, next_state = sample(hyperstate, action, info)
                next_hyperstate = update_posterior(hyperstate, action,
                    next_state, info)
                params = update_params(params, hyperstate, action,
                    next_hyperstate, reward, branching_factor)

                if next_hyperstate not in params['visited']:
                    params['U'][next_hyperstate] = info['V_max']
                    params['L'][next_hyperstate] = info['V_min']

        params = bellman_backup(hyperstate, params, info)
        
    action = params['U_a'][hyperstate]
    next_hyperstate = max(
        params['children'][(hyperstate, action)], 
        key=lambda next_hs: (params['U'][next_hs] - params['L'][next_hs]) * params['count'][(hyperstate, action)][next_hs])

    params = fsss_rollout(next_hyperstate, max_depth - 1, branching_factor, params, info)
    params = bellman_backup(hyperstate, params, info)
    return params


def fsss(hyperstate, max_depth, n_trajectories, branching_factor, params, info):
    for _ in range(n_trajectories):
        params = fsss_rollout(hyperstate, max_depth, branching_factor, params, info)
    return params['U'][hyperstate], params


def bfs3(hyperstate, max_depth, n_trajectories, branching_factor, params, info):
    if hyperstate in params['policy']:
        return params['policy'][hyperstate], params

    q_values = np.zeros(info['n_actions'], dtype=np.float32)
    for action in range(info['n_actions']):
        for _ in range(branching_factor):
            reward, next_state = sample(hyperstate, action, info)
            next_hyperstate = update_posterior(hyperstate, action,
                next_state, info)
            next_value, params = fsss(next_hyperstate, max_depth, n_trajectories, branching_factor, params, info)
            q_values[action] += (reward + info['gamma'] * next_value) / branching_factor

    params['policy'][hyperstate] = np.argmax(q_values)
    return params['policy'][hyperstate], params


def init_params(info):
    return {
        'policy': {},
        'count': {},
        'children': {},
        'R': {},
        'U': {},
        'L': {},
        'U_a': {},
        'L_a': {},
        'visited': set(),
    }


class BFS3Agent:
    def __init__(self, max_depth, n_trajectories, branching_factor, info):
        self.max_depth = max_depth
        self.n_trajectories = n_trajectories
        self.branching_factor = branching_factor
        self.info = info
        self.params = init_params(info)

    def act(self, hyperstate):
        action, self.params = bfs3(hyperstate, self.max_depth, self.n_trajectories,
            self.branching_factor, self.params, self.info)
        return action


if __name__ == "__main__":
    import argparse
    import gym

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', '-d', type=int, default=5,
        help='Maximum planning depth.')
    parser.add_argument('--n-trajectories', '-t', type=int, default=10,
        help='Number of trajectories considered per FSSS.')
    parser.add_argument('--branching-factor', '-b', type=int, default=30,
        help='Number of branches created per decision when using BFS3/FSSS.')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='Discount factor.')
    args = parser.parse_args()

    env = gym.make('NChain-v0')
    info = get_info(env, args)
    
    print('Enviroment Information:\n {}'.format(info))

    agent = BFS3Agent(args.max_depth, args.n_trajectories, args.branching_factor, info)

    rsum, t = 0, 0
    state, done = env.reset(), False
    hyperstate = init_hyperstate(state, info)
    while not done:
        action = agent.act(hyperstate)
        next_state, rew, done, _ =  env.step(action)
        hyperstate = update_posterior(hyperstate, action, next_state, info)
        print(rew)
        rsum += rew
        t += 1
    print(rsum, t, rsum / t)
