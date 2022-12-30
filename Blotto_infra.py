from math import comb
from abc import ABC, abstractmethod
import numpy as np
import random
from numpy import cumsum
from numpy.random import rand
from collections.abc import Iterable
from infrastructure import *

def soldier_dist(i, N, S, out):
    if S == 0:
        out += [0] * N
        return np.array(out)
    if N == 1:
        out.append(S)
        return np.array(out)

    temp = 1
    for s in range(S + 1):
        if s >= 1:
            temp = (temp * (N + s - 2)) // s
        if temp > i:
            out.append(S - s)
            return soldier_dist(i, N - 1, s, out)
        else:
            i -= temp

class Blotto(Game):
    def __init__(self, N, S_arr):
        self.N = N
        self.S_arr = S_arr
        self.player_count = 2
        self.action_counts = [comb(N + self.S_arr[i] -1, N-1) for i in range(self.player_count)]
        self.soldier_dists = [[soldier_dist(i, self.N, self.S_arr[j], []) for i in range(self.action_counts[j])] 
                              for j in range(self.player_count)]
        self.strategy_maps = [{i:f"{self.soldier_dists[j][i]}" for i in 
                                range(self.action_counts[j])} for j in range(self.player_count)]
        self._setup()
    
    def _utility(self, index: int, *actions: Iterable) -> float:
        dists = [self.soldier_dists[i][actions[i]] for i in range(self.player_count)]
        if index == 0:
            return sum(dists[0] > dists[1]) - sum(dists[0] < dists[1])
        else:
            return -self._utility(0, *actions)

    