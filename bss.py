from collections import namedtuple
import random

import gym
import numpy as np


class Node:
    def __init__(self, type, belstate, depth, edge=None):
        self.type = type
        self.belstate = belstate
        self.depth = depth
        self.edge = edge
        self.children = {}


def get_info(env):
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
    }


def _get_index(state, action, next_state, info):
    n_states = info['n_states']
    n_actions = info['n_actions']
    return (state * n_states * n_actions +
            action * n_states +
            next_state)


def init_belstate(state, info, prior_alphas=None):
    n_params = info['n_states'] * info['n_actions'] * info['n_states']
    if not prior_alphas:
        prior_alphas = np.ones(n_params, dtype=np.long)
    counts = np.zeros(n_params, dtype=np.long)
    return (prior_alphas, state, counts)


def sample(belstate, action, info):
    alphas, state, _ = belstate

    # sample next state
    start = _get_index(state, action, 0, info)
    relevant_alphas = alphas[start:start + info['n_states']]
    transition_probas = np.random.dirichlet(relevant_alphas)
    next_state = np.argmax(np.random.multinomial(1, transition_probas))
    
    # get corresponding reward
    reward = info['reward_fn'](state, action, next_state)

    return reward, state


def update_posterior(belstate, action, next_state, info):
    alphas, state, counts = belstate

    idx = _get_index(state, action, next_state, info)
    new_alphas = alphas.copy()
    new_alphas[idx] += 1
    new_counts = counts.copy()
    new_counts[idx] += 1

    return (new_alphas, next_state, new_counts)


def thompson_sample(belstate, info):
    alphas, state, _ = belstate

    best_action, best_q_value = None, float('-inf')
    for action in range(info['n_actions']):
        # sample model
        start = _get_index(state, action, 0, info)
        relevant_alphas = alphas[start:start + info['n_states']]
        transition_probas = np.random.dirichlet(relevant_alphas)

        # compute myoptic q value
        q_value = 0
        for next_state in range(info['n_states']):
            q_value += transition_probas[next_state] * info['reward_fn'](state, action, next_state)

        # keep track of best q value
        if q_value > best_q_value:
            best_q_value, best_action = q_value, action
    
    return best_action


def max_expected_reward(belstate, info):
    alphas, state, _ = belstate

    max_rew, best_action = float('-inf'), None
    for action in range(info['n_actions']):
        start = _get_index(state, action, 0, info)
        relevant_alphas = alphas[start:start + info['n_states']]
        transition_probas = relevant_alphas / np.sum(relevant_alphas) # take mean instead of sampling

        # compute mean reward
        mean_rew = 0
        for next_state in range(info['n_states']):
            mean_rew += transition_probas[next_state] * info['reward_fn'](state, action, next_state)

        # keep track of best q value
        if  mean_rew > max_rew:
            max_rew, best_action = mean_rew, action
    
    return max_rew, best_action


def eval_tree(root, horizon, info):
    if not root.children:
        assert root.type == 'decision', 'All leaf nodes are decision nodes'
        immed, best_action = max_expected_reward(root.belstate, info)
        return immed * (horizon - root.depth), best_action

    if root.type == 'decision':
        best_q_value, best_action = float('-inf'), None
        for action, node in root.children.items():
            q_value, _ = eval_tree(node, horizon, info)
            if q_value > best_q_value:
                best_q_value, best_action = q_value, action
        return best_q_value, best_action

    if root.type == 'outcome':
        estimates = []
        for key, node in root.children.items():
            rew, _ = key
            estimates.append(rew + eval_tree(node, horizon, info)[0])
        return sum(estimates) / len(estimates), None


def grow_sparse_bayesian_tree(root, budget, branch_proba, horizon, info):
    for _ in range(budget):
        branch_node, edge_value = bayes_descent(root, branch_proba, info)

        if branch_node.depth >= horizon:
            continue

        if branch_node.type == 'decision':
            action = edge_value

            # add outcome node
            outcome_node = Node('outcome', branch_node.belstate, branch_node.depth, action)
            branch_node.children[action] = outcome_node

            # add decision node
            rew, next_state = sample(branch_node.belstate, action, info)
            posterior = update_posterior(branch_node.belstate, action, next_state, info)

            decision_node = Node('decision', posterior, branch_node.depth + 1, (rew, next_state))
            outcome_node.children[(rew, next_state)] = decision_node
    
        elif branch_node.type == 'outcome':
            # add decision node
            rew, next_state = edge_value
            posterior = update_posterior(branch_node.belstate, action, next_state, info)

            decision_node = Node('decision', posterior, branch_node.depth + 1, (rew, next_state))
            branch_node.children[(rew, next_state)] = decision_node
    
    return eval_tree(root, horizon, info)


def bayes_descent(node, branch_proba, info):
    if node.type == 'decision':
        a = thompson_sample(node.belstate, info)
        if a not in node.children: # expand tree
            return node, a
        else: # go deeper
            return bayes_descent(node.children[a], branch_proba, info)

    if node.type == 'outcome':
        rew, next_state = sample(node.belstate, node.edge, info)
        if (random.random() < branch_proba
            or (rew, next_state) not in node.children): # with probability branch_proba, start a new branch here
            return node, (rew, next_state)
        
        # otherwise, follow
        return bayes_descent(node.children[(rew, next_state)], branch_proba, info)


def bss_planning(belstate, budget, branch_proba, horizon, info):
    root = Node('decision', belstate, 0)
    return grow_sparse_bayesian_tree(root, budget, branch_proba, horizon, info)


if __name__ == "__main__":
    import gym

    env = gym.make('NChain-v0')
    info = get_info(env)
    
    print(info)

    state = env.reset()
    root = Node('decision', init_belstate(state, info), 0)
    value = grow_sparse_bayesian_tree(root, 10, 0.0, 10, info)

    print(value)

    def print_tree(root, name='r'):
        print('{}\n'
              '  type: {}\n'
              '  counts: {}\n'
              '  depth: {}\n'
              '  edge: {}'.format(name, root.type, root.belstate[2], root.depth, root.edge))
        for i, c in enumerate(root.children.values()):
            print_tree(c, name=name+str(i))

    print_tree(root)

    rsum, t = 0, 0
    state, done = env.reset(), False
    belstate = init_belstate(state, info)
    while not done:
        value, action = bss_planning(belstate, 500, 0.01, 15, info)
        #action = env.action_space.sample()
        next_state, rew, done, _ =  env.step(action)
        belstate = update_posterior(belstate, action, next_state, info)
        print(rew)
        rsum += rew
        t += 1

    print(rsum, t, rsum / t)

