from collections import namedtuple
import random

import gym
import numpy as np


class Node:
    def __init__(self, type, hyperstate, depth, edge=None):
        self.type = type
        self.hyperstate = hyperstate
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


def init_hyperstate(state, info, prior_alphas=None):
    n_params = info['n_states'] * info['n_actions'] * info['n_states']
    if not prior_alphas:
        prior_alphas = np.ones(n_params, dtype=np.long)
    counts = np.zeros(n_params, dtype=np.long)
    return (prior_alphas, state, counts)


def sample(hyperstate, action, info):
    alphas, state, _ = hyperstate

    # sample next state
    start = _get_index(state, action, 0, info)
    relevant_alphas = alphas[start:start + info['n_states']]
    transition_probas = np.random.dirichlet(relevant_alphas)
    next_state = np.argmax(np.random.multinomial(1, transition_probas))
    
    # get corresponding reward
    reward = info['reward_fn'](state, action, next_state)

    return reward, state


def update_posterior(hyperstate, action, next_state, info):
    alphas, state, counts = hyperstate

    idx = _get_index(state, action, next_state, info)
    new_alphas = alphas.copy()
    new_alphas[idx] += 1
    new_counts = counts.copy()
    new_counts[idx] += 1

    return (new_alphas, next_state, new_counts)


def thompson_sample(hyperstate, info):
    alphas, state, _ = hyperstate

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


def max_expected_reward(hyperstate, info):
    alphas, state, _ = hyperstate

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
    """
    Evaluates (computes state value estimates) the bayesian sparse subtree 
    rooted at 'root'. Returns both the state value and the optimal action for
    decision nodes.
    """
    if not root.children:
        assert root.type == 'decision', 'All leaf nodes are decision nodes'
        immed, best_action = max_expected_reward(root.hyperstate, info)
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
    """
    Grows a bayesian sparse tree from a root node by adding 'budget' nodes
    according to the Bayesian Sparse Sampling algorithm.
    Returns the evaluation of the subtree starting at 'root'.
    """
    num_nodes = 1
    while num_nodes < budget:
        branch_node, edge_value = bayes_descent(root, branch_proba, info)

        if branch_node.depth >= horizon:
            continue

        if branch_node.type == 'decision':
            action = edge_value

            # add outcome node
            outcome_node = Node('outcome', branch_node.hyperstate, branch_node.depth, action)
            branch_node.children[action] = outcome_node

            # add decision node
            rew, next_state = sample(branch_node.hyperstate, action, info)
            posterior = update_posterior(branch_node.hyperstate, action, next_state, info)

            decision_node = Node('decision', posterior, branch_node.depth + 1, (rew, next_state))
            outcome_node.children[(rew, next_state)] = decision_node

            num_nodes += 2
    
        elif branch_node.type == 'outcome':
            # add decision node
            rew, next_state = edge_value
            posterior = update_posterior(branch_node.hyperstate, action, next_state, info)

            decision_node = Node('decision', posterior, branch_node.depth + 1, (rew, next_state))
            branch_node.children[(rew, next_state)] = decision_node

            num_nodes += 1
    
    return eval_tree(root, horizon, info)


def bayes_descent(node, branch_proba, info):
    """
    Descends on an exisiting bayesian sparse tree and chooses a new node to add
    to the tree.
    """
    if node.type == 'decision':
        a = thompson_sample(node.hyperstate, info)
        if a not in node.children: # expand tree
            return node, a
        else: # go deeper
            return bayes_descent(node.children[a], branch_proba, info)

    if node.type == 'outcome':
        rew, next_state = sample(node.hyperstate, node.edge, info)
        if (random.random() < branch_proba
            or (rew, next_state) not in node.children): # with probability branch_proba, start a new branch here
            return node, (rew, next_state)
        
        # otherwise, go deeper
        return bayes_descent(node.children[(rew, next_state)], branch_proba, info)


class BSSAgent:
    def __init__(self, budget, branch_proba, horizon, info):
        self.budget = budget
        self.branch_proba = branch_proba
        self.horizon = horizon
        self.info = info

    def act(self, hyperstate):
        root = Node('decision', hyperstate, 0)
        value, action = grow_sparse_bayesian_tree(root, self.budget,
            self.branch_proba, self.horizon, self.info)
        return action, {'value': value, 'root': root}


if __name__ == "__main__":
    import argparse
    import gym

    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', '-b', type=int, default=200,
        help='Number of nodes to explore per planning step.')
    parser.add_argument('--branch-proba', '-bp', type=float, default=0.0,
        help='Probability to create a new random branch on an outcome node.')
    parser.add_argument('--horizon', '-h', type=int, default=5,
        help='Maximum planning depth.')
    args = parser.parse_args()

    env = gym.make('NChain-v0')
    info = get_info(env)
    
    print('Enviroment Information:\n {}'.format(info))

    agent = BSSAgent(args.budget, args.branch_proba, args.horizon, info)

    rsum, t = 0, 0
    state, done = env.reset(), False
    hyperstate = init_hyperstate(state, info)
    while not done:
        action, _ = agent.act(hyperstate)
        next_state, rew, done, _ =  env.step(action)
        hyperstate = update_posterior(hyperstate, action, next_state, info)
        rsum += rew
        t += 1
    print(rsum, t, rsum / t)

