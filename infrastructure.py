from abc import ABC, abstractmethod
import numpy as np
import random
from numpy import cumsum
from numpy.random import rand
from collections.abc import Iterable
from typing import List

class Game(ABC):
    player_count = None
    action_counts = None
    strategy_maps = None
    """
    Encodes information about the game.

    Attributes:
    ---------------
        player_count:
            number of players in the game
        action_count:
            number of possible actions
    """
    @abstractmethod
    def __init__(self):
        pass
    
    def _setup(self):
        self.utility = np.vectorize(self._utility, excluded=['self', 'index'])
        self.payoff_matrix:Iterable = [self.utility(i, *np.meshgrid(
                                        *(range(self.action_counts[j]) for j in range(self.player_count)), indexing='ij'))
                                         for i in range(self.player_count)]

    def _get_utility(self, index, actions):
        return self.payoff_matrix[index][actions]

    def _utility(self, index: int, *actions: Iterable) -> float:
        """
        Args:
        ----------------------
        action:
            action[i] = action of player i
        index:
            index of player
        
        Output:
        ----------------------
            Utility of player

        """
        pass

class BimatrixGame(Game):
    def __init__(self, A, B):
        self.player_count = 2
        self.payoff_matrix = [A, B]
        self.action_counts = np.shape(A)    

class Agent(ABC):
    def __init__(self, game: Game, index: int):
        """
        Encodes information about a player.

        Attributes:
        --------------
        game:
            Game object repersenting what game the agent is playing
        index:
            A label for the agent
        strategy:
            A vector of weights for each action. The agent will act according to this array on each round
        total_utility:
            Total utility for the agent throughout the game.
        """
        self.game: Game = game
        self.action_count = game.action_counts[index]
        self.index:int = index
        self.strategy: Iterable = np.ones(self.action_count) / self.action_count
        self.strategy_cumsum = np.cumsum(self.strategy)
        self.total_utility: float = 0
        self.cum_strat = None
        
        
    def action(self) -> int:
        """
        Returns the action of the agent.
        """
        return random.choices(range(self.action_count), cum_weights=self.strategy_cumsum)[0]

    def update(self, actions: Iterable):
        """
        Updates the 
        """
        chosen_utility = self.game._get_utility(self.index, actions)
        self.total_utility += chosen_utility
        return chosen_utility

class Regret_Minimisation_Agent(Agent):
    def __init__(self, game: Game, index: int):
        super().__init__(game, index)
        self.regrets:np.array = np.zeros(self.action_count)
        self.strategy_sum: np.array = np.copy(self.strategy)
    
    def update(self, actions: tuple):
        chosen_utility = super().update(actions)
        
        for i in range(self.action_count):
            temp = list(actions)
            temp[self.index] = i
            temp = tuple(temp)
            action_i_utility: float = self.game._get_utility(self.index, temp)
            self.regrets[i] += action_i_utility - chosen_utility
        # self.strategy = np.array([i if i >= 0 else 0 for i in self.regrets])
        self.strategy = np.maximum(0, self.regrets)
        normalising_const = sum(self.strategy)
        
        if normalising_const <=0:
            self.strategy = np.ones(self.action_count) / self.action_count
        else:
            self.strategy /= normalising_const
        self.strategy_sum += self.strategy
        self.strategy_cumsum = np.cumsum(self.strategy)

class Trainer:
    def __init__(self, game, agents):
        self.game = game
        self.agents = agents
    
    def train(self, n, out=True):
        for i in range(n):
            action = tuple(agent.action() for agent in self.agents)
            for agent in self.agents:
                agent.update(action)
        strats = []
        for agent in self.agents:
            temp = agent.strategy_sum / sum(agent.strategy_sum)
            agent.cum_strat = temp
            strats.append(temp)
            if out:
                print(f"Utility of Agent {agent.index} is {agent.total_utility}")
                print(np.round(temp, 3))
        return self.agents

class Evaluation():
    def __init__(self, game: Game, rounds, sample_size) -> None:
        self.sample_size = sample_size
        self.rounds = rounds
        self.agents = [find_CE(game, iterations=rounds) for i in range(sample_size)]
        self.cum_probs = np.array([np.array([agent.cum_strat for agent in agent_arr]) 
                                   for agent_arr in self.agents] )
        self.average_strategy = np.average(self.cum_probs, axis= (0, 1))
        self.sds = np.std(self.cum_probs, axis=0)
        self.game = game

class Single_Evaluation(Evaluation):
    def __init__(self, game: Game, rounds=100000) -> None:
        super().__init__(game, rounds, 1)
        self.agents: List[Agent] = self.agents[0]
        self.agent_count = len(self.agents)

    def viable_strategies(self, eps=0.001, dps=3):
        for index, agent in enumerate(self.agents):
            print(f"Viable strategies for agent {index}:")
            for action_index, prob in enumerate(np.round(agent.cum_strat, dps)):
                if prob >= eps:
                    print(f"\t Probability of playing {self.game.strategy_maps[index][action_index]} is {prob}.")
            print("\n\n\n")
        
def train_repeatedly(game: Game, each_train, sample_size):
    strats = np.empty((0, game.player_count, game.action_count))
    
    for i in range(sample_size):
        strats = np.r_[strats, [find_CE(game, iterations=each_train)]]
    
    for j in range(game.player_count):
        print(f"Standard deviation of strategy of player {j} is {np.sqrt(strats[:, j].var(axis=0))}")

def find_CE(game: Game, iterations=100000):
    agents = [Regret_Minimisation_Agent(game, i) for i in range(game.player_count)]
    return Trainer(game, agents).train(iterations, False)

if __name__ == "__main__":
    pass