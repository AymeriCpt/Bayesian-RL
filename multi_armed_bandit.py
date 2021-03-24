# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 22:35:09 2021

@author: Aymeric Capitaine
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from scipy.stats import beta

##############################################################################
# /!\ /!\: you may need to restart your Kernel between each animation (either
#plot_priors or plot_regret) for it to appear 
##############################################################################


###############################################################################
#Class for Beta-Bernoulli environment
###############################################################################

class MABEnvironment():
    
    def __init__(self, **kwargs):

        # number of arms
        self.n = len(kwargs)
        
        # Create a 2D dictionary storing the value of theta as well as its Beta prior's parameters (alpha_k, beta_k).
        # Initially, all Beta parameters are set to (1/2, 1/2) (Jeffreys prior)
        self._theta_params = {}
        
        for key, value in kwargs.items():
            assert (value >= 0) and (value <= 1), 'you must choose theta in [0,1]'
            self._theta_params[key, 'theta'] = value
            self._theta_params[key, 'prior'] = [.5, .5]
        
        
    def get_reward(self, action):
        '''
        Draws a reward in {0,1} according to a Bernouilli(theta_k), where k \in {0,...,n} is the choosen action.

        Parameters
        ----------
        action : INT
            An integer between 0 and n_arms. 
        Returns
        -------
        (reward, theta): INT, STR
            A tuple whose first argument is the reward (either 0 or 1), and the second the corresponding theta. 
        '''
        assert action <= self.n, 'There are only {} arms!'.format(self.n)
        
        _choosen_theta = 'theta' + str(action)
        _reward = np.random.binomial(n = 1, p = self._theta_params[_choosen_theta, 'theta'])
        
        return _reward, _choosen_theta
    
    def update_prior(self, reward):
        '''
        Updates the prior according to the reward obtained at step t
        
        Parameters
        ----------
        reward : INT, STR
            Output of the get_reward method
        Returns
        -------
        With x_t being the action taken at step t and r_t the corresponding reward,
        updates the prior according to the following rule:
        (alpha_k, beta_k) = (alpha_k, beta_k) + (r_t, 1 - r_t)

        '''
        _initial_params = self._theta_params[reward[-1], 'prior']
        _change = [reward[0], 1 - reward[0]]        
        #update alpha_k and beta_k
        self._theta_params[reward[-1], 'prior'] = [sum(x) for x in zip(_initial_params, _change)]
        
        return self

###############################################################################
#Class for a Thomson/greedy agent
###############################################################################

class MABAgent():
    
    def __init__(self, n_arms):
        
        self.n_arms = n_arms
        self.strategies = ['greedy','thomson', 'eps_dithering']
        
    def greedy_strategy(self, theta_params):
        '''
        Applies a greedy strategy, which consists in maximizing the immediate reward based on previous observations of the environment
        
        Parameters
        ----------
        theta_params: DICT
        A dictionary of the form: {'theta1, value':..., 'theta1, prior':[alpha_1, beta_1], 'theta2':...}
        (Expects the _theta_params attribute of MABEnvironment)
        
        Returns
        -------
        _action: INT
        the action picked following a greedy strategy

        '''
        _hat_theta = []
    
    # fill a list with expected returns of each arm (= alpha / alpha + beta)
        for k in range(self.n_arms):
            _alpha, _beta = theta_params['theta'+str(k), 'prior']
            _hat_theta.append(_alpha / (_alpha + _beta))
            
    # if all returns are the same, pick a random action
        if _hat_theta.count(_hat_theta[0]) == len(_hat_theta):
            _action = np.random.randint(low = 0, high = len(_hat_theta))
            
    # else pick the action yielding the highest return
        else:
            _action = np.argmax(_hat_theta)
            
        return _action
    
    def eps_dithering_strategy(self, theta_params, epsilon = .1):
        '''
        Applies epsilon-dithering strategy

        Parameters
        ----------
        theta_params: DICT
        A dictionary of the form: {'theta1, value':..., 'theta1, prior':[alpha_1, beta_1], 'theta2':...}
        (Expects the _theta_params attribute of MABEnvironment)
        
        epsilon: FLOAT
        A tuning parameter between 0 and 1 giving the probability of applying random action
        Returns
        -------
        _action: the action picked following the epsilon dithering strategy
        '''
        
        assert (epsilon > 0) and (epsilon < 1), "pick epsilon in ]0,1["
        
        # create a dummy variable which is equal to 1 with probability epsilon. In this case, 
        # applies random search
        _random = np.random.binomial(n = 1, p = epsilon)
        
        _action = ((1 - _random) * self.greedy_strategy(theta_params)) +  (_random * (np.random.randint(low = 0, high = self.n_arms)))
        
        return _action
    
    
    def thomson_strategy(self, theta_params):
        '''
        Applies Thomson sampling

        Parameters
        ----------
        theta_params: DICT
        A dictionary of the form: {'theta1, value':..., 'theta1, prior':[alpha_1, beta_1], 'theta2':...}
        (Expects the _theta_params attribute of MABEnvironment)

        Returns
        -------
        _action: the action picked following a Thomson sampling strategy
        '''
    
        _hat_theta = []
        # fill a list with thetas sampled from the prior
        for k in range(self.n_arms):
            _alpha, _beta = theta_params['theta'+str(k), 'prior']
            _hat_theta.append(np.random.beta(a = _alpha, b = _beta))
        # pick the action yielding the highest return
        _action = np.argmax(_hat_theta)
        
        return _action

###############################################################################
#Class for simulation
###############################################################################    

class MABSimulation():
    
    def __init__(self, ClassEnvironment, ClassAgent, x_grid = np.linspace(0,1,150), T = 50):
        
        self.T = T
        self.env = ClassEnvironment
        self.agent = ClassAgent
        self.x_grid = x_grid
        self.fig = plt.figure()
    
    def reset(self):
        '''Reset the initial state'''
        for k in range(self.env.n):
            self.env.__dict__['_theta_params']['theta' + str(k) , 'prior'] = [.5 , .5]
    
    def game_update(self, strategy = None):
        '''
        Applies an action according to 'strategy', draws the corresponding
        reward and updates the environment

        Parameters
        ----------
        strategy : STR, optional
            Either "thomson" "eps_dithering" or "greedy". None leads to an AssertError.

        Returns
        -------
        Updated simulation

        '''
        # retrieve the initial state
        _state = self.env._theta_params
        # make a move
        if strategy == "greedy":
            _action = self.agent.greedy_strategy(_state)
        elif strategy == "thomson":
            _action = self.agent.thomson_strategy(_state)
        elif strategy == 'eps_dithering':
            _action = self.agent.eps_dithering_strategy(_state)
            
        # derive a reward
        _reward = self.env.get_reward(_action)
        # update the state
        self.env.update_prior(_reward)
        
    
    def plot_priors(self, strategy = None, n_stop = 100):
        '''
        Creates an animation of the distributions over rewards throughout the
        simulation. 

        Parameters
        ----------
        strategy : STR, optional
            Either "thomson" "eps_dithering" or "greedy". None leads to an AssertError.
        n_stop : INT, optional
            number of frames for the animation (set n_stop << T if T is very high, since
            the game reaches an equilibrium quite swiftly for every strategy)
        Returns
        -------
        An animation

        '''
        assert strategy in self.agent.strategies, "Choose either 'thomson', 'eps-dithering' or 'greedy' for strategy"
        
        # initiate axis and curves
        ax = plt.axes(xlim = (0,1), ylim = (0,15))
        lines = [plt.plot([], [], label = r'$\theta_{{ {j} }}$ (= {val})'.format(j = j, val = self.env._theta_params['theta' + str(j), 'theta']))[0] for j in range(self.env.n)]
        ax.legend()
        plt.suptitle('Evolution of the reward distributions')
        plt.title('{} strategy'.format(strategy))
        
        # inner function for the initialization of the FunctionAnimation method
        def _init():
            for line in lines:
                line.set_data([], []) 
            return lines
        
        # inner function for the update step of the FunctionAnimaiton method
        def _animate(i):
            self.game_update(strategy)
            for k, line in enumerate(lines):
                _p = self.env._theta_params['theta'+str(k), 'prior']
                y_grid = beta.pdf(self.x_grid, a = _p[0], b = _p[-1])
                line.set_data(self.x_grid, y_grid)
        
        # creates and returns the animation
        _anim = animation.FuncAnimation(self.fig, _animate,
                                        init_func = _init,
                                        frames = n_stop)
        return _anim    
    
    def average_regret(self, n_exp = 100, strategy = None, resolution = 20):
        '''
        Computes the per-period regret at time t for n_exp experiences:
            Regret_t = max_{k} theta_k - theta_{x_t}
        (Where x_t is the action taken at time t), and return the average
        per-period regret.
        
        Parameters
        ----------
        n_exp : INT, optional
            Number of experiences over which the regret is averaged at each
            time t.
        strategy : STR, optional
            Either "thomson" "eps_dithering" or "greedy". None leads to an AssertError.
        resolution: INT, optional 
            number of steps between two records of the regret
        Returns
        -------
        _mean_regret : A T-long sequence containing the average per period
        regret at each step.

        '''
        assert strategy in self.agent.strategies, "Choose either 'thomson', 'eps-dithering' or 'greedy' for strategy"
        
        assert self.T % resolution == 0, "choose a number of steps which is a multiple of *resolution* "
        
        # create a (n_exp x _n_records) array containing the oracle reward
        _n_records = int(self.T / resolution)
        _oracle = np.max([self.env._theta_params['theta'+str(k) , 'theta'] for k in range(self.env.n)])
        _array_oracle = np.full(shape = (n_exp, _n_records), fill_value = _oracle)
        
        # create a (n_exp x T) array to be filled with the average rewards      
        _array_reward = np.zeros_like(_array_oracle)
        
        
        #fill the reward array:
           
        for exp in range(n_exp):
            _row = []
            for t in range(self.T):
                _s = self.env._theta_params
                if strategy == 'greedy':
                    _a = self.agent.greedy_strategy(_s)
                elif strategy == 'thomson':
                    _a = self.agent.thomson_strategy(_s)
                elif strategy == 'eps_dithering':
                    _a = self.agent.eps_dithering_strategy(_s)
                
                #record the regret with resolution:
                if t % resolution == 0:
                    _row.append(self.env._theta_params['theta' + str(_a) , 'theta'])
                    
                _r = self.env.get_reward(_a)
                self.env.update_prior(_r)
   
            _array_reward[exp , :] = _row
            self.reset()

            
        #compute the regret array
        _array_regret = _array_oracle - _array_reward
        # retrieve the column-wise mean
        _mean_regret = _array_regret.mean(axis = 0)
        
        return _mean_regret
    
    def plot_regret(self, regret_thomson, regret_eps_dithering, regret_greedy):
        '''
        Produces an animation of the average regret for both Thomson and greedy
        strategies over time

        Parameters
        ----------
        regret_thomson : 
            A flat array containing the average regrets for Thomson.
            Expect an average_regret() output
        regret_eps_dithering :
            A flat array containing the average regrets for dithering.
            Expect an average_regret() output
        regret_greedy :
            A flat array containing the average regrets for greedy.
            Expect an average_regret() output

        Returns
        -------
        An animation of the regret

        '''
        _strategies = {'greedy': regret_greedy, 'eps_dithering': regret_eps_dithering, 'thomson': regret_thomson}
        
        # initiate axis and curves
        _length_seq = regret_greedy.shape[0]
        ax = plt.axes(xlim = (0,_length_seq), ylim = (0,.3))
        lines = [plt.plot([], [], label = '{}'.format(stj))[0] for stj in self.agent.strategies]
        ax.legend()
        plt.suptitle('Average per-period regret')
        plt.title('{} steps'.format(self.T))
        
        # inner function for the initialization of the FunctionAnimation method
        def _init_regret():
            for line in lines:
                line.set_data([], []) 
            return lines
        
        # inner function for the update step of the FunctionAnimation method
        def _animate_regret(i):
            for stj, line in zip(self.agent.strategies, lines):
                x_grid = np.arange(i)
                y_grid = _strategies[stj][:i]
                line.set_data(x_grid, y_grid)
        
        # creates and returns the animation
        _anim = animation.FuncAnimation(self.fig, _animate_regret,
                                        init_func = _init_regret,
                                        frames = _length_seq)
        return _anim    
        
              

###############################################################################
# Example
###############################################################################        
              
if __name__ == '__main__':
    
    #instanciate a Bernoulli-Beta environment
    env = MABEnvironment(
        theta0 = .1,
        theta1 = .59, 
        theta2 = .6,
        theta3 = .8)
    
    # instanciate an agent
    agent = MABAgent(n_arms = 4)
    
    # instanciate a simulation
    simul = MABSimulation(env, agent, T = 2000)
    
    
    # show animation of distributions updates with Thomson sampling
    anim_thomson = simul.plot_priors("thomson")
    simul.reset()
    
    #writergif = animation.PillowWriter(fps=15) 
    #anim_thomson.save('anim_thomson.gif', writer = writergif)
    
    
    # show animation of distributions updates with a greedy strategy
    anim_greedy = simul.plot_priors("greedy") 
    simul.reset()
    
    # show animation of distributoins with an eps-dithering strategy
    anim_dithering = simul.plot_priors("eps_dithering") 
    simul.reset()

    
    
    #plot the mean regret (over 200 experiences):
    thomson_regret = simul.average_regret(n_exp = 200, strategy = "thomson")
    simul.reset()
    greedy_regret = simul.average_regret(n_exp = 200, strategy = "greedy")
    simul.reset()
    dithering_regret = simul.average_regret(n_exp = 200, strategy = "eps_dithering")
    simul.reset()
    
    anim_regret = simul.plot_regret(thomson_regret, dithering_regret, greedy_regret)
    #writergif = animation.PillowWriter(fps=25) 
    #anim_regret.save('anim_regret2.gif', writer = writergif)


        
       