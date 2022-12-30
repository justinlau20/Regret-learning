from infrastructure import *

class RPS(Game):
    player_count = 2
    strategy_maps = [{0:"Rock", 1:"Paper", 2:"Scissors"}] * player_count
    def __init__(self):
        self.action_counts = [3, 3]
        self._setup()

    def _utility(self, index: int, *actions: Iterable) -> float:
        if index == 0:
            a, b = actions
        else:
            b, a = actions
        diff = (a - b) % 3

        if diff == 2:
            return -1
        
        elif diff == 0:
            return 0
        return 1

class RPS_Rock_Agent(Agent):
    def action(self):
        return 0

class RPS_Rock_Double(RPS):
    def __init__(self):
        super().__init__()
    
    def _utility(self, index: int, *actions: Iterable) -> float:
        if index == 0:
            a, b = actions
        else:
            b, a = actions
        diff = (a - b) % 3
        
        if diff == 2:
            if actions[index] == 2:
                return -2
            return -1
        elif diff == 0:
            return 0
        if actions[index] == 0:
            return 2
        return 1

class RPS_With_Stone(RPS):
    strategy_maps = [{0:"Rock", 1:"Paper", 2:"Scissors", 3:"Rock but different label"}] * 2
    def __init__(self):
        self.action_counts = [4, 4]
        self.player_count = 2
        self._setup()
    
    def _utility(self, index: int, *actions: Iterable) -> float:
        actions = (i if i != 3 else 0 for i in actions )
        return super()._utility(index, *actions)

