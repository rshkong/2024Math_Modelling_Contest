import mesa

# Data visualization tools.
import seaborn as sns

# Has multi-dimensional arrays and matrices. Has a large collection of
# mathematical functions to operate on these arrays.
import numpy as np

# Data manipulation and analysis.
import pandas as pd
import random


r_p=2
r_m=0.75
a=0.02
b=0.8



def A(size):
  return 1/1000*size+5/1000


def reproduce_distribution():
  return random.choices([0, 1, 2, 3, 4], weights=[0.1, 0.2, 0.4, 0.2, 0.1], k=1)[0]


def S(p):
    return 3/(10*(1+np.exp(0.25*(p-28))))+0.5

class EFAgent(mesa.Agent):
    """A agent with a fixed size that grows until it reaches a maximum size."""
    def __init__(self, unique_id, model, max_size=10):
        super().__init__(unique_id, model)
        self.size = 1
        self.max_size = max_size

    def step(self):
        # Size grows until it reaches max size
        if self.size < self.max_size:
            self.size += 1
        
        # Chance of reproducing
        self.reproduce_or_die()

        self.move()


    def reproduce_or_die(self):
        # Find a neighboring cell
        d = a*self.get_surrounding_agents_count() #death probability
        if r_p>d:
            e = r_p-d   #expectation to get new babies
            num_new_agents = np.random.poisson(e)      #use poisson distribution
            for _ in range(num_new_agents):
                empty_cells = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=5)
                if empty_cells:
                    new_pos = random.choice(empty_cells)
                    new_agent = EFAgent(self.model.next_id(), self.model, self.max_size)
                    self.model.grid.place_agent(new_agent, new_pos)
                    self.model.schedule.add(new_agent)
        else:
            p = min(1,d-r_p)
            if random.random() < p:
                self.model.grid.romove_agent(self)
                self.model.schedule.remove(self)

            
    def move(self):
        # Choose a random distance to move
        distance = random.randint(0, 10)  # Random distance between 0 and 10
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=distance)
        new_position = random.choice(possible_steps) if possible_steps else self.pos
        # Move the agent
        self.model.grid.move_agent(self, new_position)

    def get_surrounding_agents_count(self, a=10):
    # a is the half-width of the square area, making the full width 2a+1
    # Get the neighborhood cells within a distance 'a' of M/F type
        neighborhood_positions = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True,radius=a)
        
        # Initialize count
        agent_count = 0
        
        # Iterate over the neighborhood positions
        for pos in neighborhood_positions:
            # Get agents in each cell
            cell_agents = self.model.grid.get_cell_list_contents([pos])
            
            # Add the number of agents in the current cell to the total count
            agent_count += sum(1 for agent in cell_agents if isinstance(agent, (MAgent,FAgent)))
        
        return agent_count-1


class DFAgent(mesa.Agent):
    """A agent with a fixed size that grows until it reaches a maximum size, but does not die or reproduce."""
    def __init__(self, unique_id, model, max_size=8):
        super().__init__(unique_id, model)
        self.size = 1
        self.max_size = max_size

    def step(self):
        if self.size < self.max_size:
            self.size += 1

        self.move()
        # Chance of reproducing
        self.reproduce_or_die()

    def reproduce_or_die(self):
        # Find a neighboring cell
        d = (a/2)*self.get_surrounding_agents_count() #death probability
        if r_p>d:
            e = r_p-d   #expectation to get new babies
            num_new_agents = np.random.poisson(e)      #use poisson distribution
            for _ in range(num_new_agents):
                empty_cells = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=5)
                if empty_cells:
                    new_pos = random.choice(empty_cells)
                    new_agent = EFAgent(self.model.next_id(), self.model, self.max_size)
                    self.model.grid.place_agent(new_agent, new_pos)
                    self.model.schedule.add(new_agent)
        else:
            p = min(1,d-r_p)
            if random.random() < p:
                self.model.grid.romove_agent(self)
                self.model.schedule.remove(self)


    def move(self):
        # Choose a random distance to move
        distance = random.randint(0, 10)  # Random distance between 0 and 10
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=distance)
        new_position = random.choice(possible_steps) if possible_steps else self.pos
        # Move the agent
        self.model.grid.move_agent(self, new_position)

    def get_surrounding_agents_count(self, a=10):
    # a is the half-width of the square area, making the full width 2a+1
    # Get the neighborhood cells within a distance 'a' of M/F type
        neighborhood_positions = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True,radius=a)
        
        # Initialize count
        agent_count = 0
        
        # Iterate over the neighborhood positions
        for pos in neighborhood_positions:
            # Get agents in each cell
            cell_agents = self.model.grid.get_cell_list_contents([pos])
            
            # Add the number of agents in the current cell to the total count
            agent_count += sum(1 for agent in cell_agents if isinstance(agent, (MAgent,FAgent)))
        
        return agent_count-1


class FAgent(mesa.Agent):
    """A simple agent type F."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        # FAgent的行为定义
        self.move()
        self.reproduce_or_die()
        


    def reproduce_or_die(self):
        M=self.model.agent_counts["MAgent"]
        F=self.model.agent_counts["FAgent"]
        near_Prey = self.get_surrounding_agents_count()
        g = min(1,M/F)*b*a*near_Prey        #growth rate
        if g>r_m:
            e = g-r_m
            num_new_agents = np.random.poisson(e)
            num_new_M = np.round(num_new_agents*S(near_Prey)).astype(int)
            num_new_F = np.round(num_new_agents*(1-S(near_Prey))).astype(int)
            empty_cells = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=5)
            for _ in range(num_new_M):
                if empty_cells:
                    new_pos = random.choice(empty_cells)
                    new_agent = MAgent(self.model.next_id(), self.model)
                    self.model.grid.place_agent(new_agent, new_pos)
                    self.model.schedule.add(new_agent)

            for _ in range(num_new_F):
                 if empty_cells:
                    new_pos = random.choice(empty_cells)
                    new_agent = FAgent(self.model.next_id(), self.model)
                    self.model.grid.place_agent(new_agent, new_pos)
                    self.model.schedule.add(new_agent)
        else:
            d = r_m-g      #death rate
            p = min(1,d-r_m)
            if random.random() < p:
                self.model.grid.romove_agent(self)
                self.model.schedule.remove(self)



    def get_surrounding_agents_count(self, a=10):
    # a is the half-width of the square area, making the full width 2a+1
    # Get the neighborhood cells within a distance 'a' of M/F type
        neighborhood_positions = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True,radius=a)
        
        # Initialize count
        agent_count = 0
        
        # Iterate over the neighborhood positions
        for pos in neighborhood_positions:
            # Get agents in each cell
            cell_agents = self.model.grid.get_cell_list_contents([pos])
            
            # Add the number of agents in the current cell to the total count
            agent_count += sum(1 for agent in cell_agents if isinstance(agent, (EFAgent,DFAgent)))
        
        return agent_count-1


    def move(self):
        # Choose a random distance to move
        distance = random.randint(0, 10)  # Random distance between 0 and 10
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=distance)
        new_position = random.choice(possible_steps) if possible_steps else self.pos
        # Move the agent
        self.model.grid.move_agent(self, new_position)


class MAgent(mesa.Agent):
    """A simple agent type M."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        # MAgent的行为定义
        self.move()
        if random.random()<r_m:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)



    def move(self):
        # Choose a random distance to move
        # distance = random.randint(0, 10)  # Random distance between 0 and 10
        a = self.pos
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=10)
        new_position = random.choice(possible_steps) if possible_steps else self.pos
        # Move the agent
        self.model.grid.move_agent(self, new_position)





class MyModel(mesa.Model):
    def __init__(self, N1,N2,N3,N4, width=100, height=100):
        super().__init__()
        self.num_EF = N1
        self.num_DF = N2
        self.num_M = N3
        self.num_F = N4
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.agent_counts = {}  # 用于存储各种代理的数量

        
        # Create agents
        for i in range(self.num_EF):
          agent = EFAgent(self.next_id(),self,max_size=10)
          x = random.randrange(self.grid.width)
          y = random.randrange(self.grid.height)
          self.grid.place_agent(agent, (x, y))
          self.schedule.add(agent)
        for i in range(self.num_DF):
          agent = DFAgent(self.next_id(),self,max_size=8)
          x = random.randrange(self.grid.width)
          y = random.randrange(self.grid.height)
          self.grid.place_agent(agent, (x, y))
          self.schedule.add(agent)
        for i in range(self.num_M):
          agent = MAgent(self.next_id(),self)
          x = random.randrange(self.grid.width)
          y = random.randrange(self.grid.height)
          self.grid.place_agent(agent, (x, y))
          self.schedule.add(agent)
        for i in range(self.num_F):
          agent = FAgent(self.next_id(),self)
          x = random.randrange(self.grid.width)
          y = random.randrange(self.grid.height)
          self.grid.place_agent(agent, (x, y))
          self.schedule.add(agent)

        self.update_agent_counts()  # 初始化代理数量信息

    def update_agent_counts(self):
        # 更新模型中各种代理的数量
        self.agent_counts = {'EFAgent': 0, 'DFAgent': 0, 'FAgent':0, 'MAgent':0}  # 重置计数
        for agent in self.schedule.agents:
            agent_type = type(agent).__name__
            if agent_type in self.agent_counts:
                self.agent_counts[agent_type] += 1

    def step(self):
        self.update_agent_counts()  # 在每个时间步更新代理数量信息
        self.schedule.step()



model=MyModel(10,10,10,10)
for i in range(100):
    model.step()