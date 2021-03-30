from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col, translate, adjacent_positions, min_distance
import random as rand
from enum import Enum, auto


def opposite(action):
    if action == Action.NORTH:
        return Action.SOUTH
    if action == Action.SOUTH:
        return Action.NORTH
    if action == Action.EAST:
        return Action.WEST
    if action == Action.WEST:
        return Action.EAST
    raise TypeError(str(action) + " is not a valid Action.")

    

#Enconding of cell content to build states from observations
class CellState(Enum):
    EMPTY = 0
    FOOD = auto()
    GOOSE = auto()


#This class encapsulates mos of the low level Hugry Geese stuff    
class BornToNotMedalv2:    
    def __init__(self):
        self.DEBUG=True
        self.rows, self.columns = -1, -1        
        self.my_index = -1
        self.my_head, self.my_tail = -1, -1
        self.geese = []
        self.heads = []
        self.tails = []
        self.food = []
        self.cell_states = []
        self.actions = [action for action in Action]
        self.previous_action = None
        self.step = 1

        
    def _adjacent_positions(self, position):
        return adjacent_positions(position, self.columns, self.rows)
 

    def _min_distance_to_food(self, position, food=None):
        food = food if food!=None else self.food
        return min_distance(position, food, self.columns)

    
    def _row_col(self, position):
        return row_col(position, self.columns)
    
    
    def _translate(self, position, direction):
        return translate(position, direction, self.columns, self.rows)
        
        
    def preprocess_env(self, observation, configuration):
        observation = Observation(observation)
        configuration = Configuration(configuration)
        
        self.rows, self.columns = configuration.rows, configuration.columns        
        self.my_index = observation.index
        self.hunger_rate = configuration.hunger_rate
        self.min_food = configuration.min_food

        self.my_head, self.my_tail = observation.geese[self.my_index][0], observation.geese[self.my_index][-1]        
        self.my_body = [pos for pos in observation.geese[self.my_index][1:-1]]

        
        self.geese = [g for i,g in enumerate(observation.geese) if i!=self.my_index  and len(g) > 0]
        self.geese_cells = [pos for g in self.geese for pos in g if len(g) > 0]
        
        self.occupied = [p for p in self.geese_cells]
        self.occupied.extend([p for p in observation.geese[self.my_index]])
        
        
        self.heads = [g[0] for i,g in enumerate(observation.geese) if i!=self.my_index and len(g) > 0]
        self.bodies = [pos  for i,g in enumerate(observation.geese) for pos in g[1:-1] if i!=self.my_index and len(g) > 2]
        self.tails = [g[-1] for i,g in enumerate(observation.geese) if i!=self.my_index and len(g) > 1]
        self.food = [f for f in observation.food]
        
        self.adjacent_to_heads = [pos for head in self.heads for pos in self._adjacent_positions(head)]
        self.adjacent_to_bodies = [pos for body in self.bodies for pos in self._adjacent_positions(body)]
        self.adjacent_to_tails = [pos for tail in self.tails for pos in self._adjacent_positions(tail)]
        self.adjacent_to_geese = self.adjacent_to_heads + self.adjacent_to_bodies
        self.danger_zone = self.adjacent_to_geese
        
        #Cell occupation
        self.cell_states = [CellState.EMPTY.value for _ in range(self.rows*self.columns)]
        for g in self.geese:
            for pos in g:
                self.cell_states[pos] = CellState.GOOSE.value
        for pos in self.heads:
                self.cell_states[pos] = CellState.GOOSE.value
        for pos in self.my_body:
            self.cell_states[pos] = CellState.GOOSE.value
                
        #detect dead-ends
        self.dead_ends = []
        for pos_i,_ in enumerate(self.cell_states):
            if self.cell_states[pos_i] != CellState.EMPTY.value:
                continue
            adjacent = self._adjacent_positions(pos_i)
            adjacent_states = [self.cell_states[adj_pos] for adj_pos in adjacent if adj_pos!=self.my_head]
            num_blocked = sum(adjacent_states)
            if num_blocked>=(CellState.GOOSE.value*3):
                self.dead_ends.append(pos_i)
        
        #check for extended dead-ends
        new_dead_ends = [pos for pos in self.dead_ends]
        while new_dead_ends!=[]:
            for pos in new_dead_ends:
                self.cell_states[pos]=CellState.GOOSE.value
                self.dead_ends.append(pos)
            
            new_dead_ends = []
            for pos_i,_ in enumerate(self.cell_states):
                if self.cell_states[pos_i] != CellState.EMPTY.value:
                    continue
                adjacent = self._adjacent_positions(pos_i)
                adjacent_states = [self.cell_states[adj_pos] for adj_pos in adjacent if adj_pos!=self.my_head]
                num_blocked = sum(adjacent_states)
                if num_blocked>=(CellState.GOOSE.value*3):
                    new_dead_ends.append(pos_i)                                    
        
                
    def strategy_random(self, observation, configuration):
        if self.previous_action!=None:
            action = rand.choice([action for action in Action if action!=opposite(self.previous_action)])
        else:
            action = rand.choice([action for action in Action])
        self.previous_action = action
        return action.name
                        
                        
    def safe_position(self, future_position):
        return (future_position not in self.occupied) and (future_position not in self.adjacent_to_heads) and (future_position not in self.dead_ends)
    
    
    def valid_position(self, future_position):
        return (future_position not in self.occupied) and (future_position not in self.dead_ends)    

    
    def free_position(self, future_position):
        return (future_position not in self.occupied) 
    
                        
    def strategy_random_avoid_collision(self, observation, configuration):
        dead_end_cell = False
        free_cell = True
        actions = [action 
                   for action in Action 
                   for future_position in [self._translate(self.my_head, action)]
                   if self.valid_position(future_position)] 
        if self.previous_action!=None:
            actions = [action for action in actions if action!=opposite(self.previous_action)] 
        if actions==[]:
            dead_end_cell = True
            actions = [action 
                       for action in Action 
                       for future_position in [self._translate(self.my_head, action)]
                       if self.free_position(future_position)]
            if self.previous_action!=None:
                actions = [action for action in actions if action!=opposite(self.previous_action)] 
            #no alternatives
            if actions==[]:
                free_cell = False
                actions = self.actions if self.previous_action==None else [action for action in self.actions if action!=opposite(self.previous_action)] 

        action = rand.choice(actions)
        self.previous_action = action
        if self.DEBUG:
            aux_pos = self._row_col(self._translate(self.my_head, self.previous_action))
            dead_ends = "" if not dead_end_cell else f', dead_ends={[self._row_col(p1) for p1 in self.dead_ends]}, occupied={[self._row_col(p2) for p2 in self.occupied]}'
            if free_cell:
                print(f'{id(self)}({self.step}): Random_ac_move {action.name} to {aux_pos} dead_end={dead_end_cell}{dead_ends}', flush=True)
            else:
                print(f'{id(self)}({self.step}): Random_ac_move {action.name} to {aux_pos} free_cell={free_cell}', flush=True)
        return action.name
    
    
    def strategy_greedy_avoid_risk(self, observation, configuration):        
        actions = {  
            action: self._min_distance_to_food(future_position)
            for action in Action 
            for future_position in [self._translate(self.my_head, action)]
            if self.safe_position(future_position)
        }
  
        if self.previous_action!=None:
            actions.pop(opposite(self.previous_action), None)
        if any(actions):
            action = min(actions.items(), key=lambda x: x[1])[0]
            self.previous_action = action
            if self.DEBUG:
                aux_pos = self._row_col(self._translate(self.my_head, self.previous_action))
                print(f'{id(self)}({self.step}): Greedy_ar_move {action.name} to {aux_pos}', flush=True)
            self.previous_action = action
            return action.name
        else:
            return self.strategy_random_avoid_collision(observation, configuration)
    
    
    #Redefine this method
    def agent_strategy(self, observation, configuration):
        action = self.strategy_greedy_avoid_risk(observation, configuration)
        return action
    
    
    def agent_do(self, observation, configuration):
        self.preprocess_env(observation, configuration)
        move = self.agent_strategy(observation, configuration)
        self.step += 1
        #if self.DEBUG:
        #    aux_pos = self._translate(self.my_head, self.previous_action), self._row_col(self._translate(self.my_head, self.previous_action))
        #    print(f'{id(self)}({self.step}): Move {move} to {aux_pos} internal_vars->{vars(self)}', flush=True)
        return move

    
    
def agent_singleton(observation, configuration):
    global gus    
    
    try:
        gus
    except NameError:
        gus = BornToNotMedalv2()
            
    action = gus.agent_do(observation, configuration)

    
    return action