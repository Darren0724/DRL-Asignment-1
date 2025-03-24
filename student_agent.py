import numpy as np
import pickle
import random

# Load the trained Q-table
with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

# Global variables
move_history = {}  # (row, col, action) -> True if visited
visit_count = {}   # (row, col) -> number of visits
now_doing = 0
goal_r = -1
goal_c = -1
now_r = 0
now_c = 0
row = [0]*4
col = [0]*4
st = -1
ed = -1
last_action = 0
epsilon = 0.05  # Initial exploration rate
alpha = 0.1
gamma = 0.99
rec_reward = 0
rec_state = None
step = 0

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def get_state_key(obs):
    global move_history, now_r, now_c
    obstacle_north = obs[10]
    obstacle_south = obs[11]
    obstacle_east = obs[12]
    obstacle_west = obs[13]
    
    north_state = 1 if obstacle_north else (2 if (now_r, now_c, 1) in move_history else 0)
    south_state = 1 if obstacle_south else (2 if (now_r, now_c, 0) in move_history else 0)
    east_state = 1 if obstacle_east else (2 if (now_r, now_c, 2) in move_history else 0)
    west_state = 1 if obstacle_west else (2 if (now_r, now_c, 3) in move_history else 0)
    
    return (north_state, south_state, east_state, west_state, sign(goal_r - now_r), sign(goal_c - now_c))

def get_action(obs, reward=None, next_obs=None):
    global now_doing, goal_r, goal_c, now_r, now_c, row, col, st, ed, last_action, q_table
    global move_history, visit_count, rec_reward, rec_state, step, epsilon
    
    step += 1
    
    # Dynamic epsilon adjustment
    if step <= 50:
        epsilon = 0.05
    elif step <= 100:
        epsilon = 0.1
    elif step <= 150:
        epsilon = 0.15
    else:
        epsilon = min(0.15 + (step - 150) * 0.001, 0.2)  # Gradual increase, cap at 0.2
    
    # Online learning if reward is provided
    if rec_reward is not None and rec_state is not None:
        next_state = get_state_key(obs)
        if next_state not in q_table:
            q_table[next_state] = np.zeros(6)
        q_table[rec_state][last_action] += alpha * (
            rec_reward + gamma * np.max(q_table[next_state]) - q_table[rec_state][last_action]
        )
    
    # Update positions
    for i in range(4):
        row[i] = obs[2*i+2]
        col[i] = obs[2*i+3]
    now_r = obs[0]
    now_c = obs[1]
    
    # Initialize goal
    if goal_r == -1:
        goal_r = row[0]
        goal_c = col[0]
        move_history.clear()
        visit_count.clear()
    
    # State machine
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
    
    # Define valid actions
    valid_actions = [i for i in range(4)]
    if now_doing == 5 and now_r == goal_r and now_c == goal_c:
        valid_actions = [4]
    elif now_doing == 7 and now_r == goal_r and now_c == goal_c:
        valid_actions = [5]
    
    # Get current state
    state = get_state_key(obs)
    if state not in q_table:
        q_table[state] = np.zeros(6)
    q_values = q_table[state]
    
    # Increase exploration if stuck (based on visit count)
    current_pos = (now_r, now_c)
    visit_count[current_pos] = visit_count.get(current_pos, 0) + 1
    if visit_count[current_pos] > 5:  # If visited too many times, boost exploration
        local_epsilon = min(epsilon + 0.1, 0.3)
    else:
        local_epsilon = epsilon
    
    # ε-greedy action selection
    if random.random() < local_epsilon:
        last_action = random.choice(valid_actions)
    else:
        last_action = max(valid_actions, key=lambda a: q_values[a])
    
    # Update move history
    if last_action in [0, 1, 2, 3] and not (last_action == 0 and obs[11] or last_action == 1 and obs[10] or last_action == 2 and obs[12] or last_action == 3 and obs[13]):
        move_history[(now_r, now_c, last_action)] = True
    
    # Consistent reward shaping (aligned with your previous train design)
    shaped_reward = 0
    dir1 = sign(goal_r - now_r)
    dir2 = sign(goal_c - now_c)
    
    if last_action == 0:  # South
        if now_r == goal_r and now_c == goal_c:
            shaped_reward += 5
        if obs[11] == 1:
            shaped_reward -= 100
        if dir1 == 1:
            shaped_reward += 1
        elif dir1 == -1:
            shaped_reward -= 1
        if not obs[11] and (now_r, now_c, 0) in move_history:
            shaped_reward -= 0.5
    elif last_action == 1:  # North
        if now_r == goal_r and now_c == goal_c:
            shaped_reward += 5
        if obs[10] == 1:
            shaped_reward -= 100
        if dir1 == -1:
            shaped_reward += 1
        elif dir1 == 1:
            shaped_reward -= 1
        if not obs[10] and (now_r, now_c, 1) in move_history:
            shaped_reward -= 0.5
    elif last_action == 2:  # East
        if now_r == goal_r and now_c == goal_c:
            shaped_reward += 5
        if obs[12] == 1:
            shaped_reward -= 100
        if dir2 == 1:
            shaped_reward += 1
        elif dir2 == -1:
            shaped_reward -= 1
        if not obs[12] and (now_r, now_c, 2) in move_history:
            shaped_reward -= 0.5
    elif last_action == 3:  # West
        if now_r == goal_r and now_c == goal_c:
            shaped_reward += 5
        if obs[13] == 1:
            shaped_reward -= 100
        if dir2 == -1:
            shaped_reward += 1
        elif dir2 == 1:
            shaped_reward -= 1
        if not obs[13] and (now_r, now_c, 3) in move_history:
            shaped_reward -= 0.5
    elif last_action == 4:  # PICKUP
        if now_doing == 5 and now_r == goal_r and now_c == goal_c:
            shaped_reward += 10
        else:
            shaped_reward -= 100
    elif last_action == 5:  # DROPOFF
        if now_doing == 7 and now_r == goal_r and now_c == goal_c:
            shaped_reward += 10
        else:
            shaped_reward -= 100
    
    # Update for next iteration
    rec_reward = shaped_reward
    rec_state = state
    
    return last_action

if __name__ == "__main__":
    print("Q-table loaded with size:", len(q_table))