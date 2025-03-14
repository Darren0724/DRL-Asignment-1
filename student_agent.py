# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    #env.get_state()
    taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    if obstacle_south == 0:
        return 0
    if obstacle_north == 0:
        return 1
    if obstacle_east == 0:
        return 2
    if obstacle_west == 0:
        return 3
    return random.choice([0, 1, 2, 3]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

