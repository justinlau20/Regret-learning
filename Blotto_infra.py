from math import comb
from abc import ABC, abstractmethod
import numpy as np
import random
from numpy import cumsum
from numpy.random import rand
from collections.abc import Iterable
from infrastructure import *

def partition_matrix(k, s):
    """
    K: Maxmimal number of partitions
    S: Number to be partitioned

    Output:
        Returns a (K+1) by (S+1) matrix whose i,j th entry is the number of ways to partition the number j
        into the sum of i non-negative numbers where the order does not matter 
    """
    out = np.array([np.array([1] + [0] * s)], dtype=int)
    for i in range(1, k + 1):
        mul = np.zeros(s + 1, dtype=int)
        for j in range(len(mul)):
            if (j - len(mul) + 1) % i == 0:
                mul[j] = 1
        prod_out = np.polymul(mul, np.flip(out[-1, :]))
        row = np.flip(prod_out)[:s + 1]
        out = np.r_[out, [row]]
    return out

def soldier_dist(i, N, S, out):
    if S == 0:
        out += [0] * N
        return np.array(out, dtype=int)
    if N == 1:
        out.append(S)
        return np.array(out, dtype=int)
    temp = 1
    for s in range(S + 1):
        if s >= 1:
            temp = (temp * (N + s - 2)) // s
        if temp > i:
            out.append(S - s)
            return soldier_dist(i, N - 1, s, out)
        else:
            i -= temp

def soldier_dist_collapsed(i, N, S, lookup_mat, out, base=0):
    if S == 0:
        out += [base] * N
        return np.array(out)

    if N == 1:
        out.append(base + S)
        return np.array(out)
    
    for s in range(S + 1):
        temp = lookup_mat[N - 1, S - s * N]
        if temp > i:
            base += s
            out.append(base)
            return soldier_dist_collapsed(i, N - 1, S - s * N, lookup_mat, out, base)
        else:
            i -= temp
    raise Exception("You are bad at coding >:(")

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

def average_payoff(A, B):
    s = 0

    for a in A:
        diff = a - B
        s += sum(diff > 0) - sum(diff < 0)

    return s / len(A) 

class Collapsed_Blotto(Game):
    def __init__(self, N, S_arr):
        self.N = N
        self.S_arr = S_arr
        self.player_count = 2
        if len(set(S_arr)) == 1:
            self.lookup_mats = [partition_matrix(self.N, self.S_arr[0])] * 2
        else:
            self.lookup_mats = [partition_matrix(self.N, self.S_arr[i]) for i in self.S_arr]

        self.action_counts = [self.lookup_mats[i][-1, -1] for i in range(self.player_count)]

        self.soldier_dists = [[soldier_dist_collapsed(i, self.N, self.S_arr[j], self.lookup_mats[j], []) 
                              for i in range(self.action_counts[j])] for j in range(self.player_count)]

        self.strategy_maps = [{i:f"{self.soldier_dists[j][i]}" for i in 
                                range(self.action_counts[j])} for j in range(self.player_count)]

    def _get_utility(self, index, actions):
        dists = [self.soldier_dists[i][actions[i]] for i in range(self.player_count)]
        random.shuffle(dists[0])

        d = dists[0] - dists[1]
        temp = sum(d > 0) - sum(d < 0)

        if index == 0:
            return temp
        else:
            return -temp

# Single_Evaluation(Collapsed_Blotto(3, (7, 7)), 10000).viable_strategies(eps=0.01)
    