import gym
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
import random

class SimpleTaxiEnv:
    def __init__(self, grid_size=5, fuel_limit=50):
        """
        Custom Taxi environment with ~10% of cells as obstacles.
        """
        self.grid_size = grid_size
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False
        
        self.stations = [
            (0, 0),  # R
            (0, self.grid_size - 1),  # G
            (self.grid_size - 1, 0),  # Y
            (self.grid_size - 1, self.grid_size - 1)  # B
        ]
        self.passenger_loc = None
        self.obstacles = set()  # Will be populated in reset
        self.destination = None

    def reset(self):
        """Reset the environment with ~10% obstacles."""
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False
        
        # All possible positions
        all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        
        # Calculate number of obstacles (~10% of grid size)
        total_cells = self.grid_size * self.grid_size
        num_obstacles = max(1, round(total_cells * 0.1))  # At least 1 obstacle
        
        # Choose taxi position first
        available_positions = [pos for pos in all_positions if pos not in self.stations]
        self.taxi_pos = random.choice(available_positions)
        
        # Choose passenger and destination
        self.passenger_loc = random.choice([pos for pos in self.stations])
        possible_destinations = [s for s in self.stations if s != self.passenger_loc]
        self.destination = random.choice(possible_destinations)
        
        # Exclude taxi, passenger, and destination from obstacle placement
        excluded_positions = {self.taxi_pos, self.passenger_loc, self.destination}
        available_for_obstacles = [pos for pos in all_positions if pos not in excluded_positions and pos not in self.stations]
        
        # Place obstacles
        self.obstacles = set(random.sample(available_for_obstacles, min(num_obstacles, len(available_for_obstacles))))
        
        return self.get_state(), {}

    def step(self, action):
        """Perform an action and update the environment state with reward shaping."""
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0

        # Calculate current distances
        passenger_dist = abs(taxi_row - self.passenger_loc[0]) + abs(taxi_col - self.passenger_loc[1])
        dest_dist = abs(taxi_row - self.destination[0]) + abs(taxi_col - self.destination[1])

        if action == 0:  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1

        if action in [0, 1, 2, 3]:  # Movement actions
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -= 5  # Penalty for hitting obstacle or wall
            else:
                # Update position
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos

                # Calculate new distances after move
                new_passenger_dist = abs(next_row - self.passenger_loc[0]) + abs(next_col - self.passenger_loc[1])
                new_dest_dist = abs(next_row - self.destination[0]) + abs(next_col - self.destination[1])

                # Reward shaping
                if not self.passenger_picked_up:
                    if new_passenger_dist < passenger_dist:
                        reward += 2  # Small reward for moving closer to passenger
                    elif new_passenger_dist > passenger_dist:
                        reward -= 1  # Small penalty for moving away from passenger
                else:
                    if new_dest_dist < dest_dist:
                        reward += 2  # Small reward for moving closer to destination
                    elif new_dest_dist > dest_dist:
                        reward -= 1  # Small penalty for moving away from destination

        elif action == 4:  # PICKUP
            if self.taxi_pos == self.passenger_loc and not self.passenger_picked_up:
                self.passenger_picked_up = True
                self.passenger_loc = self.taxi_pos
                reward += 50  # Base reward for successful pickup
            else:
                reward -= 10  # Penalty for incorrect pickup

        elif action == 5:  # DROPOFF
            if self.passenger_picked_up and self.taxi_pos == self.destination:
                reward += 500  # Base reward for successful dropoff
                return self.get_state(), reward - 0.1, True, {}
            elif self.passenger_picked_up:
                reward -= 10  # Penalty for incorrect dropoff
            else:
                reward -= 10  # Penalty for dropoff without passenger

        reward -= 0.1  # Small penalty for each step
        self.current_fuel -= 1
        if self.current_fuel <= 0:
            return self.get_state(), reward - 10, True, {}

        return self.get_state(), reward, False, {}

    def get_state(self):
        """Return the current environment state."""
        taxi_row, taxi_col = self.taxi_pos
        passenger_row, passenger_col = self.passenger_loc
        destination_row, destination_col = self.destination
        
        obstacle_north = int(taxi_row == 0 or (taxi_row - 1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row + 1, taxi_col) in self.obstacles)
        obstacle_east = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col + 1) in self.obstacles)
        obstacle_west = int(taxi_col == 0 or (taxi_row, taxi_col - 1) in self.obstacles)

        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle = int((taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle
       
        destination_loc_north = int((taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int((taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east = int((taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west = int((taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle = int((taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle

        state = (
            taxi_row, taxi_col,
            self.stations[0][0], self.stations[0][1],
            self.stations[1][0], self.stations[1][1],
            self.stations[2][0], self.stations[2][1],
            self.stations[3][0], self.stations[3][1],
            obstacle_north, obstacle_south, obstacle_east, obstacle_west,
            passenger_look, destination_look
        )
        return state

    def render_env(self, taxi_pos, action=None, step=None, fuel=None):
        """Render the environment with obstacles visualized."""
        clear_output(wait=True)

        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
        
        # Place stations
        grid[0][0] = 'R'
        grid[0][self.grid_size - 1] = 'G'
        grid[self.grid_size - 1][0] = 'Y'
        grid[self.grid_size - 1][self.grid_size - 1] = 'B'

        # Place obstacles
        for obs_y, obs_x in self.obstacles:
            if 0 <= obs_y < self.grid_size and 0 <= obs_x < self.grid_size:
                grid[obs_y][obs_x] = 'X'

        # Place taxi (overwrites obstacles if overlapping, but reset prevents this)
        ty, tx = taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            grid[ty][tx] = '🚖'

        # Print step info
        print(f"\nStep: {step}")
        print(f"Taxi Position: ({tx}, {ty})")
        print(f"Passenger Location: {self.passenger_loc}")
        print(f"Destination: {self.destination}")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")

        # Print grid
        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"

def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = SimpleTaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    stations = [(0, 0), (0, 4), (4, 0), (4,4)]
    
    taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    if render:
        env.render_env((taxi_row, taxi_col),
                       action=None, step=step_count, fuel=env.current_fuel)
        time.sleep(0.5)
    while not done:
        
        
        action = student_agent.get_action(obs)

        obs, reward, done, _ = env.step(action)
        print('obs=',obs)
        total_reward += reward
        step_count += 1

        taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs

        if render:
            env.render_env((taxi_row, taxi_col),
                           action=action, step=step_count, fuel=env.current_fuel)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

if __name__ == "__main__":
    env_config = {
        "fuel_limit": 5000
    }
    
    agent_score = run_agent("student_agent.py", env_config, render=True)
    print(f"Final Score: {agent_score}")