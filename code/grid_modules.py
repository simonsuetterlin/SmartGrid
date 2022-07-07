#from rk4step import rk4step
import numpy as np
import casadi as ca
import random

class GridModules:
    def __init__(self, name: str, adapt_speed):
        self.name = name
        self.bla = None
        self.adapt_speed = adapt_speed
        self.price = 10
        self.simulation = None

    
def Consumer(GridModules):
    def __init__(self, **kwargs):
        super.__init__(**kwargs)
        self.current_consumption = None

    def consumption(self, time: float=0):
        '''
        Calculates how much power is currently drawn from the input from the grid.
        
        Args: 
            time (float): current time point 
        
        Return(int, float): current input
        '''
        assert 0 <= time and time < 1, 'time must be between 0 and 1.'
        return self.current_consumption
    
def Generator(GridModules):
    def __init__(self, **kwargs):
        super.__init__(**kwargs)
        self.current_prod = None
        
    def production(self, time: float=0):
        '''
        Calculates how much power is currently given as input to the grid.
        
        Args: 
            decision (int): change of speed 
        
        Return(int, float): current input
        '''
        assert 0 <= time and time < 1, 'time must be between 0 and 1.'
        return self.current_prod + self.adapt_speed(t) * self.simulation.decision
    
    def loss(self, time: float=0):
        '''
        Calculates how much it costs to produce current input to the grid.
        
        Args: 
             (int): change of speed 
        
        Return(int, float): current input
        '''
        assert 0 <= time and time < 1, 'time must be between 0 and 1.'
        return self.current_prod + self.adapt_speed(t)

    
def Storage(GridModules):

# Consumer, speicher, Erzeuger