import numpy as np
import pickle
import random 
from time import sleep
with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

now_doing = 0 # 0: moving1, 1: moving2, 2: moving3, 3: moving4, 4: move pickup, 5: pickup, 6: move dropoff 7: dropoff
goal_r = -1
goal_c = -1
now_r = 0
now_c = 0
row = [0]*4
col = [0]*4
st = -1
ed = -1
last_action = 0
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
    # Extract only the surrounding wall info from the full observation
    obstacle_north = obs[10]
    obstacle_south = obs[11]
    obstacle_east = obs[12]
    obstacle_west = obs[13]
    return (now_doing, obstacle_north, obstacle_south, obstacle_east, obstacle_west, sign2(goal_r - now_r), sign2(goal_c - now_c))


def get_action(obs):
    state = get_state_key(obs)
    global now_doing,goal_r,goal_c ,now_r ,now_c ,row ,col ,st ,ed ,last_action,q_table
    #print(now_doing,st,ed)
    #print(goal_c,goal_r,now_c,now_r)
    #print(state)
    #if state not in q_table:
    #    print('not in table')
    #else:
    #    print(q_table[state])
    #print(obs)
    #print(last_action)
    #sleep(0.5)
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
    elif now_doing == 4 :
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
    if state not in q_table:
        valid_actions = [i for i in range(4)] 
        if now_doing == 5:
            valid_actions.append(4)
        elif now_doing == 7:
            valid_actions.append(5) 
        last_action = random.choice(valid_actions) if valid_actions else random.randint(0, 5)
        return last_action
    last_action = np.argmax(q_table[state])
    return last_action
if __name__ == "__main__":
    print(q_table)