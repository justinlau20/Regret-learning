from abc import ABC, abstractmethod
import numpy as np
import random
from numpy import cumsum
from numpy.random import rand
class Game(ABC):
    def __init__(self):
        self.player_count = 0
        self.action_count = 0
        
    @abstractmethod
    def utility(self, action):
        """
        Args:
        ----------------------
        action:
            array; action[i] = action of player i
        
        Output:
        ----------------------
            arr[i] = utility of player i
        """
        pass

class Agent(ABC):
    def __init__(self, game: Game, index):
        """
        Agents can keep track of certain statistics, such as regret.
        """
        self.game = game
        self.index = index
        self.strategy = np.ones(game.action_count) / game.action_count
        self.total_utility = 0
        
        
    def action(self):
        return random.choices(range(self.game.action_count), weights=self.strategy)[0]
    
    def update(self, actions):
        """
        Takes as input the actions of all agents in the round
        """
        chosen_utility = self.game.utility(actions = actions, index = self.index)
        self.total_utility += chosen_utility
        return chosen_utility

class Regret_Minimisation_Agent(Agent):
    def __init__(self, game, index):
        super().__init__(game, index)
        self.regrets = np.zeros(self.game.action_count)
        self.strategy_sum = np.copy(self.strategy)
    
    def update(self, actions):
        chosen_utility = super().update(actions)
        
        for i in range(self.game.action_count):
            temp = list(actions)
            temp[self.index] = i
            action_i_utility = self.game.utility(temp, self.index)
            self.regrets[i] += action_i_utility - chosen_utility
            
        self.strategy = np.array([i if i >= 0 else 0 for i in self.regrets ])
        normalising_const = sum(self.strategy)
        
        if normalising_const <=0:
            self.strategy = np.ones(self.game.action_count) / self.game.action_count
        else:
            self.strategy /= sum(self.strategy)
        self.strategy_sum += self.strategy

class Trainer:
    def __init__(self, game, agents):
        self.game = game
        self.agents = agents
    
    def train(self, n, out=True):
        for i in range(n):
            action = [agent.action() for agent in self.agents]
            for agent in self.agents:
                agent.update(action)

        if out:
            for agent in self.agents:
                print(f"Utility of Agent {agent.index} is {agent.total_utility}")
                if isinstance(agent, Regret_Minimisation_Agent) :
                    print(agent.strategy_sum / sum(agent.strategy_sum))

def find_Nash(game: Game, iterations=100000):
    agents = [Regret_Minimisation_Agent(game, i) for i in range(game.player_count)]
    Trainer(game, agents).train(iterations)

if __name__ == "__main__":
    pass