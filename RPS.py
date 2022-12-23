from infrastructure import *

class RPS(Game):
    def __init__(self):
        super().__init__()
        self.action_count = 3
        self.player_count = 2

    def utility(self, actions, index):
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
    
    def utility(self, actions, index):
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