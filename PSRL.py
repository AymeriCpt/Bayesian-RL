# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 11:17:14 2021

@author: Aymeric Capitaine
"""
import numpy as np 
import itertools

# define a |S|x|A| array containing the true rewards for each state-action pair
def rewards():
    _rewards = np.zeros(shape = (5, 2))
    _rewards[-1, 0] = 10
    _rewards[0, -1] = 2
    return _rewards 

# define a |A|x|S|x|S| array containing the true probabilities of transition
# from s to s' under action a
def true_probs():
    _probs = np.zeros(shape = (2, 5, 5))
    _probs[0, :, 0] = .2
    _probs[1, :, 0] = .8
    _probs[0, 4, 4] = .8
    _probs[1, 4, 4] = .2
    for dim, row, col in itertools.product(range(_probs.shape[0]),
                             range(_probs.shape[1]),
                             range(_probs.shape[2])):
        if (dim == 0) and (col == row + 1):
            _probs[dim, row, col] = .8
        elif (dim == 1) and (col == row + 1):
            _probs[dim, row, col] = .2
        else:
            pass
    return _probs
    
    
    
class ChainPSLR():

    def __init__(self, S = 5, A = 2, M = 1e3, tau = 20):
        '''
        Parameters
        ----------
        S : Cardinal of the hyper-state space
        A : Cardinal of the actino space
        M : Number of episodes for the PSRL algorithm
        tau : length of each sequence
        '''
    
        self.tau = tau
        self.S = int(S)
        self.A = int(A)
        self.M = int(M)
        self.true_probs = true_probs()
        # prior parameters initialized to 1/S
        self.alpha = np.full(shape = (self.A,self.S,self.S), fill_value = 1/self.S)
        self.rewards = rewards()
        
    def reset(self):
        self.alpha = np.full(shape = (self.A,self.S,self.S), fill_value = 1/self.S)
        return self
        
    def bellman(self, old_V, probs, rewards):
        '''
        

        Parameters
        ----------
        old_V : Sx1 array
            
        probs : TYPE
            DESCRIPTION.
        rewards : TYPE
            DESCRIPTION.

        Returns
        -------
        new_V : TYPE
            DESCRIPTION.
        new_pol : TYPE
            DESCRIPTION.

        '''
        Q_vals = np.zeros_like(rewards)
        
        for state, action in itertools.product(
                range(rewards.shape[0]),
                range(rewards.shape[-1])):
            Q_vals[state , action] = rewards[state , action] + ( probs[action, state, :] @ old_V )
        
        #find out which action maximizes the action-value function for each state
        
        #add a small perturbation in case of equality
        SMALL = 1e-8
        Q_vals = Q_vals + (SMALL * np.random.normal(size = (Q_vals.shape[0], Q_vals.shape[1])))
        
        # return max over rows of Q_vals
        new_V, new_pol =  Q_vals.max(axis = 1), (Q_vals.argmax(axis = 1)).astype(np.int)
        
        return new_V, new_pol
    
    def value_iteration(self, probs, rewards):
        
        old_V = np.zeros(shape = (5,))
        
        for _ in range(self.tau):
            new_V, policy = self.bellman(old_V, probs, rewards)
            old_V = new_V
            
        return new_V, policy
    
    def sample_dirichlet(self, alpha):
        
        output_probs = np.zeros(shape = (self.A, self.S, self.S))
        for action, state in itertools.product(range(alpha.shape[0]),
                                               range(alpha.shape[1])):
            output_probs[action, state, :] = np.random.dirichlet(self.alpha[action, state, :])
            
        return output_probs
    
    def game_update(self, current_state, policy):
        action = policy[current_state]
        reward = self.rewards[current_state , action]
        next_state = np.random.choice(self.S, p = self.true_probs[action, current_state, :])
    
        return action, next_state, reward
    
    def PSRL(self):
        
        #initialization
        #T = self.M * self.tau
        
        for episode in range(self.M):
            # at each episode, start at the left-most state
            current_state = 0
            # sample a MDP based on the game historic
            sampled_MDP = self.sample_dirichlet(self.alpha)
            # solve this MDP: derive an optimal policy by value iteration
            value, policy = self.value_iteration(sampled_MDP, self.rewards)
            print(policy)
            # follow this policy during tau steps
            for t in range(self.tau):
                action, new_state, reward = self.game_update(current_state, policy)
                
                # at the last step, update the prior's parameters
                if t == range(self.tau)[-1]:
                    self.alpha[action, current_state, new_state] += 1
                current_state = new_state
                
if __name__ == '__main__':
    
    psrl = ChainPSLR(5, 2, 500, 20)
    psrl.PSRL()
        