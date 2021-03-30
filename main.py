
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col, translate, adjacent_positions, min_distance
import random as rand
from enum import Enum, auto
import pickle
import os



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
    #search space gets too big too fast... so just 3 cell states
    #HEAD = auto()
    #BODY = auto()
    #TAIL = auto()
    
    

#This class encapsulates a simple qlearning with epsilon-greedy policy, the states and transitions can be defined automatically as we explore (search space is too big to initialize all at once and many states won't be achievable)
class QLearner():
    def __init__(self, actions, states=None, initial_value=0.1, alpha=0.3, gamma=0.1, epsilon=0.9, create_states_on_exploration=True):
        self.actions = actions
        self.create_states_on_exploration = create_states_on_exploration
        self.initial_value = initial_value
        if states!=None:
            self.q_table = {
                state: [initial_value for _ in self.actions] for state in states
            }
            self.states = states
        else:
            self.q_table = dict()
            self.states = []
            
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.previous_state = None
        self.current_state = None
        self.last_action = None
        self.last_action_index = None

        
    def _check_auto_init_state(self, state):
        if (state!=None) and (state not in self.q_table.keys()) and self.create_states_on_exploration:
            self.q_table[state] = [self.initial_value for _ in self.actions]
            self.states.append(state)
    

    def _epsilon_greedy(self, state):
        #create state if needed
        self._check_auto_init_state(state)
        
        if (rand.random() < self.epsilon):
            action = rand.choice(self.actions)
            self.last_action_index = self.actions.index(action)
        else:
            q_state = self.q_table[state]
            max_val = max(q_state)
            self.last_action_index = rand.choice([i for i,v in enumerate(q_state) if v==max_val])
            action = self.actions[self.last_action_index]
        return action

    
    def process_reward(self, reward, previous_state=None, last_action=None, last_action_index=None):
        if previous_state==None:
            previous_state = self.previous_state
        if last_action==None:
            last_action = self.last_action
        if last_action_index==None:
            last_action_index = self.last_action_index
            
        if (previous_state==None) or (last_action==None):
            return
        
        #create state if needed
        self._check_auto_init_state(previous_state)
        
        q = self.q_table
        q_old = q[previous_state][last_action_index]
        next_state = self.current_state
        if next_state!=None:        
            best_scenario = q[next_state].index(max(q[next_state]))
            q[previous_state][last_action_index] = q_old + self.alpha * (reward + self.gamma * best_scenario - q_old)
        else:
            q[previous_state][last_action_index] = q_old + self.alpha * (reward + self.initial_value - q_old)

            
    def epsilon_greedy_choose_action(self, state):
        self.previous_state = self.current_state
        self.current_state = state
        self.last_action = self._epsilon_greedy(state)
        return self.last_action
    
    
    def reset_internal_states(self):
        self.previous_state = None
        self.current_state = None
        self.last_action = None
        self.last_action_index = None
          
            
    def save_pickle(self, name):
        save_data = (self.actions,
                     self.q_table,
                     self.states,
                     )
        with open(f'{name}', 'wb') as handle:
            pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            
    def load_pickle(self, name):
        with open(f'{name}', 'rb') as handle:
            data = pickle.load(handle)
            self.actions, self.q_table, self.states = data
            
            

#This class encapsulates mos of the low level Hugry Geese stuff    
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


        
#This is our Q-Learning Hungry Geese Agent
class QGoose(BornToNotMedalv2, QLearner):    
    def __init__(self):
        self.POV_DISTANCE=3
        BornToNotMedalv2.__init__(self)
        QLearner.__init__(self, self.actions, initial_value=0.01, alpha=0.1, gamma=.8, epsilon=0.1)
        self.world = None
        self.previous_length = 0
        self.last_min_distance_to_food = self.rows*self.columns #initial max value to mark no food seen so far
       
    
    def title_state_from_row_col(self, row, col):
        pos = self.columns*row+col-1
        if pos in self.heads or pos==self.my_head:
            return CellState.GOOSE
        elif pos in self.bodies or pos==self.my_body:
            return CellState.GOOSE
        elif pos in self.tails or pos==self.my_tail:
            return CellState.GOOSE
        elif pos in self.food:
            return CellState.FOOD
        else:
            return CellState.EMPTY
    
    
    def state_from_world(self):
        state = []
        row_0, col_0 = self._row_col(self.my_head)
        for col_delta in range (-self.POV_DISTANCE, self.POV_DISTANCE):
            for row_delta in range (-self.POV_DISTANCE, self.POV_DISTANCE):
                row_i = (row_0+row_delta)%self.rows
                col_i = (col_0+col_delta)%self.columns
                state.append(self.title_state_from_row_col(row_i, col_i))
        state = "".join([str(s.value) for s in state])
        return state
    
    
    def common_sense_after_move_choosen(self, action):
        future_position = self._translate(self.my_head, action)
  
        if future_position in self.occupied:
            return -10 
        elif self.previous_action==opposite(self.last_action): #opposite is currently a patch until Action.opposite works...
            return -10
        elif self.previous_action in self.dead_ends:
            return -1
        else:
            min_distance_to_food = self._min_distance_to_food(future_position)
            aux_last = self.last_min_distance_to_food
            self.last_min_distance_to_food=min_distance_to_food
            
            if min_distance_to_food<aux_last:
                return 1
            else:
                return 0
    
    
    def agent_strategy(self, observation, configuration):
        state = self.state_from_world()
        
        #Process reward for growing
        reward = len(self.my_body)+2*self.step #Geese really like pizza!!! 8-)
        self.previous_length = len(self.my_body)
        self.process_reward(reward)
        
        #Choose action
        action = self.epsilon_greedy_choose_action(state)
        
        #Apply some common sense like colliding is bad... ;-)
        cs_reward = self.common_sense_after_move_choosen(action)
        if cs_reward<0:
            #update q-table
            self.process_reward(reward, previous_state=state, last_action=action, last_action_index=self.actions.index(action))

            #choose new greedy risk averse valid action
            random_action = self.strategy_greedy_avoid_risk(observation, configuration)                                   
            #update internal action attributes
            aux = [(action,index) for index,action in enumerate(Action) if action.name==random_action][0]
            self.last_action = aux[0]
            self.last_action_index = aux[1]
            action = self.last_action
        print(f'q-agent q_table{self.q_table}', flush=True)
        
        self.previous_action = action    
        return Action(action).name


        
def agent_singleton(observation, configuration):
    global gus    
    saved="qgoose.pickle"
    
    try:
        gus
    except NameError:
        gus = QGoose()
        if os.path.isfile(saved) and os.stat(saved).st_size>0:
            if gus.DEBUG:
                print("Loading agent, q-table...")
            gus.load_pickle(saved)
            if gus.DEBUG:
                print("Loaded!")
        elif gus.DEBUG:
            print("No previous trained QTable found!")
            
    action = gus.agent_do(observation, configuration)
    #print("Saving QGoose pickle!!!", saved)
    gus.save_pickle(saved)
    
    return action