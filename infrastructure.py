from abc import ABC, abstractmethod
import numpy as np
import random
from numpy import cumsum
from numpy.random import rand
from collections.abc import Iterable
from typing import List
import matplotlib.pyplot as plt
from plotting import *

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
        self.payoff_matrix:Iterable = np.array([self.utility(i, *np.meshgrid( 
                                        *(range(self.action_counts[j]) 
                                        for j in range(self.player_count)), 
                                        indexing='ij'))
                                         for i in range(self.player_count)])

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
    def __init__(self, game: Game, index: int, prior=None, save_strat_hist=False):
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
        self.t = 0
        self.game: Game = game
        self.action_count = game.action_counts[index]
        self.index:int = index
        if prior is None:
            self.strategy: Iterable = np.ones(self.action_count) / self.action_count
        else:
            self.strategy = prior
        self.strategy_sum: np.array = np.copy(self.strategy)
        self.strategy_cumsum = np.cumsum(self.strategy)
        self.total_utility: float = 0
        self.cum_strat = None
        self.avg_overall_regrets = []
        self.strategy_sums = np.empty((self.action_count, 0))
        self.save_strat_hist = save_strat_hist
        
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
        # print(chosen_utility)

        self.strategy_sum += self.strategy
        if self.save_strat_hist:
            self.strategy_sums = np.c_[self.strategy_sums, self.strategy_sum / sum(self.strategy_sum)]

        self.t += 1
        return chosen_utility


class Regret_Minimisation_Agent(Agent):
    def __init__(self, game: Game, index: int, prior=None):
        super().__init__(game, index, prior)
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

        self.strategy = np.maximum(0, self.regrets)
        normalising_const = sum(self.strategy)
        
        if normalising_const <=0:
            self.strategy = np.ones(self.action_count) / self.action_count
        else:
            self.strategy /= normalising_const
        # self.strategy_sum += self.strategy
        self.strategy_sum[actions[self.index]] += 1
        self.strategy_cumsum = np.cumsum(self.strategy)
        self.avg_overall_regrets.append(max(self.regrets) / self.t)


class Swap_Regret_Agent(Agent):
    def __init__(self, game: Game, index: int, prior=None):
        super().__init__(game, index, prior)
        self.swap_regret = np.zeros((self.action_count, self.action_count))
        self.cum_swap_diff = np.zeros((self.action_count, self.action_count))
        self.prev_action = None
        self.mu =  2 * (max(game.action_counts) - 1) * abs(game.payoff_matrix).max() + 1

    def update(self, actions: Iterable):
        chosen_utility = super().update(actions)
        chosen = actions[self.index]
        self.prev_action = chosen
        for i in range(self.action_count):
            temp = list(actions)
            temp[self.index] = i
            temp = tuple(temp)
            action_i_utility: float = self.game._get_utility(self.index, temp)
            self.cum_swap_diff[i][chosen] += action_i_utility - chosen_utility
        
        row = self.cum_swap_diff[:, chosen].clip(min=0)
        row[self.prev_action] = 0
        row = row / (self.t * self.mu)
        p_stay = 1 - sum(row)
        row[self.prev_action] = p_stay
        self.strategy = row
        self.strategy_sum[chosen] += 1
        self.strategy_cumsum = np.cumsum(self.strategy)
        self.avg_overall_regrets.append(self.cum_swap_diff.max() / self.t)


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
            agent.avg_utility = agent.total_utility / n
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
    def __init__(self, game: Game, rounds=100000, agent = Regret_Minimisation_Agent, priors=None, save_hist=False) -> None:
        self.rounds = rounds
        self.game = game
        self.agents: List[Agent] = find_CE(game, agent=agent, iterations=rounds, priors = priors, save_hist=save_hist)
        self.agent_count = len(self.agents)

    def viable_strategies(self, eps=0.001, dps=3):
        for index, agent in enumerate(self.agents):
            print(f"Viable strategies for agent {index}:")
            for action_index, prob in enumerate(np.round(agent.cum_strat, dps)):
                if prob >= eps:
                    if self.game.strategy_maps:
                        print(f"\t Probability of playing {self.game.strategy_maps[index][action_index] } (with index {action_index}) is {prob}.")
                    else:
                        print(f"\t Probability of playing {action_index} is {prob}.")
            print("\n\n\n")
    
    def get_strategies(self):
        return [agent.cum_strat for agent in self.agents]
    
    def plot_regrets(self, index=0):
        x_arr = np.arange(1, self.rounds)
        plt.plot(self.agents[index].avg_overall_regrets, label="Maximal regret", alpha=0.7)
        plt.plot(x_arr, 1 / np.sqrt(x_arr), label="1/sqrt(x)", alpha=0.7)
        plt.xlabel("Number of rounds played")
        
        plt.legend()
        plt.savefig("./plots/simple game/maximal_regret")
        plt.show()
        
        

def train_repeatedly(game: Game, each_train, sample_size):
    strats = np.empty((0, game.player_count, game.action_count))
    
    for i in range(sample_size):
        strats = np.r_[strats, [find_CE(game, iterations=each_train)]]
    
    for j in range(game.player_count):
        print(f"Standard deviation of strategy of player {j} is {np.sqrt(strats[:, j].var(axis=0))}")


def find_CE(game: Game, agent= Regret_Minimisation_Agent, iterations=100000, priors=None, save_hist=False):
    if priors is None:
        agents = [agent(game, i) for i in range(game.player_count)]
    else:
        agents = [agent(game, i, priors[i]) for i in range(game.player_count)]
    if save_hist:
        for agent in agents:
            agent.save_strat_hist = True
    return Trainer(game, agents).train(iterations, False)


def evaluate(agent:Agent, index=0, round=10000, mat=False):
    game = agent.game
    n = game.action_counts[1 - index]
    agents = [0, 0]
    agents[index] = agent
    m = 0

    for i in range(n):
        prior = np.zeros(n)
        prior[i] = 1
        agents[1 - index] = Agent(game, 1- index, prior)
        
        if not mat:
            
            outcome = Trainer(game, agents).train(round, out=False)
            pure_regret = outcome[1 - index].avg_utility
        else:
            cur_mat = game.payoff_matrix[1 - index]
            pure_regret =  agents[index].strategy @ cur_mat @ np.transpose(agents[1 - index].strategy)
        m = max(m, pure_regret)
    return m

def faceoff(a1, a2, round=10000, mat=True):
    game = a1.game
    agents = [a1, a2]

    # outcome = Trainer(game, agents).train(round, out=False)
    cur_mat = game.payoff_matrix[1]
    if mat:
        return agents[0].strategy @ cur_mat @ np.transpose(agents[1].strategy)
    else:
        outcome = Trainer(game, agents).train(round, out=False)
        return outcome[1].avg_utility

if __name__ == "__main__":
    pass