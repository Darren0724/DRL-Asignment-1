import numpy as np
import pickle
import random 
from time import sleep

with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

now_doing = 0  # 0: moving1, 1: moving2, 2: moving3, 3: moving4, 4: move pickup, 5: pickup, 6: move dropoff, 7: dropoff
goal_r = -1
goal_c = -1
now_r = 0
now_c = 0
row = [0]*4
col = [0]*4
st = -1
ed = -1
last_action = 0
tau = 0.1  # Fixed temperature for testing

def sign(x):   
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def sign2(x):   
    if x > 3:
        return 2
    elif x > 0:
        return 1
    elif x < -3:
        return -2
    elif x < 0:
        return -1
    else:
        return 0

def get_state_key(obs):
    obstacle_north = obs[10]
    obstacle_south = obs[11]
    obstacle_east = obs[12]
    obstacle_west = obs[13]
    return (now_doing, obstacle_north, obstacle_south, obstacle_east, obstacle_west, sign2(goal_r - now_r), sign2(goal_c - now_c))

def softmax(q_values, tau):
    """Compute Softmax probabilities for Q-values with temperature tau."""
    # Subtract max to prevent overflow
    q_values = q_values - np.max(q_values[np.isfinite(q_values)])  # Only consider finite values
    exp_q = np.exp(q_values / tau)
    sum_exp_q = np.sum(exp_q)
    if sum_exp_q == 0 or not np.isfinite(sum_exp_q):  # Handle all -inf or overflow
        # If all actions are -inf, assign uniform probability to valid actions
        probabilities = np.ones_like(q_values) / np.sum(np.ones_like(q_values))
    else:
        probabilities = exp_q / sum_exp_q
    return probabilities

def get_action(obs):
    global now_doing, goal_r, goal_c, now_r, now_c, row, col, st, ed, last_action, q_table
    state = get_state_key(obs)
    
    for i in range(4):
        row[i] = obs[2*i+2]
        col[i] = obs[2*i+3]
    if goal_r == -1:
        goal_r = row[0]
        goal_c = col[0]
    now_r = obs[0]
    now_c = obs[1]
    
    if now_doing < 4:
        if now_r == goal_r and now_c == goal_c:
            if obs[14] == 1:
                st = now_doing
            if obs[15] == 1:
                ed = now_doing
            if now_doing == 3:
                now_doing = 4
                goal_r = row[st]
                goal_c = col[st]
            else:
                now_doing += 1
                goal_r = row[now_doing]
                goal_c = col[now_doing]
    elif now_doing == 4:
        if now_r == goal_r and now_c == goal_c:
            now_doing = 5
    elif now_doing == 5:
        if now_r == goal_r and now_c == goal_c and last_action == 4:
            now_doing = 6
            goal_r = row[ed]
            goal_c = col[ed]
    elif now_doing == 6:
        if now_r == goal_r and now_c == goal_c:
            now_doing = 7
    elif now_doing == 7:
        if now_r == goal_r and now_c == goal_c and last_action == 5:
            now_doing = 8
    
    # Define valid actions based on now_doing
    valid_actions = [i for i in range(4)]  # Movement actions
    if now_doing == 5:
        valid_actions.append(4)  # Add PICKUP
    elif now_doing == 7:
        valid_actions.append(5)  # Add DROPOFF
    
    # Softmax action selection
    if state not in q_table:
        q_table[state] = np.zeros(6)  # Initialize if not in table
    q_values = q_table[state]
    action_mask = np.full(6, -np.inf)  # Mask invalid actions
    for a in valid_actions:
        action_mask[a] = q_values[a]
    probabilities = softmax(action_mask, tau)
    last_action = np.random.choice(6, p=probabilities)
    return last_action

if __name__ == "__main__":
    print(q_table)