"""
Created by:     Dex Vo
Date:           10/30/2018
Class:          CSC 371, Assignment 4

This python code generates Gridworld Two from Assignment 4.

I start off with the code from Anson Wong (git: ankonzoid) that trains
an epsilon-greedy agent to find the shortest path between two corners of
a 2D rectangular grid without any obstacles.

This code generates a 2D grid environment with wind flowing from bottom to top.
"""
import os, sys, random, operator
import numpy as np

class Environment:

    def __init__(self, Ny=7, Nx=10):
        # Define state space
        self.Ny = Ny  # y grid size
        self.Nx = Nx  # x grid size
        self.state_dim = (Ny, Nx)
        # Define action space
        self.action_dim = (4,)  # up, right, down, left
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_coords = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # translations
        # Define rewards table
        self.R = self._build_rewards()  # R(s,a) agent rewards
        # Define wind
        self.wind = self._build_wind()
        # Check action space consistency
        if len(self.action_dict.keys()) != len(self.action_coords):
            exit("err: inconsistent actions given")

    def reset(self):
        # Reset agent state to bottom left corner
        self.state = (self.Ny - 4, 0)
        return self.state

    def step(self, action):
        # Check for wind
        wind_blow = 0
        if action == self.action_dict["right"] or action == self.action_dict["left"]:
            wind_blow = self.wind[self.state[1] + self.action_coords[action][1]]
        # Evolve agent state
        state_next = (max(self.state[0] + self.action_coords[action][0] - wind_blow, 0),
                      self.state[1] + self.action_coords[action][1])
        # Collect reward
        reward = self.R[self.state + (action,)]
        # Terminate if we reach a cliff or the goal
        done = (state_next[0] == self.Ny - 4) and (state_next[1] == self.Nx - 3)
        # Update state
        self.state = state_next
        return state_next, reward, done

    def allowed_actions(self):
        # Generate list of actions allowed depending on agent grid location
        actions_allowed = []
        y, x = self.state[0], self.state[1]
        if (y > 0):  # no passing top-boundary
            actions_allowed.append(self.action_dict["up"])
        if (y < self.Ny - 1):  # no passing bottom-boundary
            actions_allowed.append(self.action_dict["down"])
        if (x > 0):  # no passing left-boundary
            actions_allowed.append(self.action_dict["left"])
        if (x < self.Nx - 1):  # no passing right-boundary
            actions_allowed.append(self.action_dict["right"])
        actions_allowed = np.array(actions_allowed, dtype=int)
        return actions_allowed

    def _build_rewards(self):
        # Define agent rewards R[s,a]
        r_goal = 100
        r_nongoal = -1
        R = r_nongoal * np.ones(self.state_dim + self.action_dim, dtype=float)
        R[self.Ny - 3, self.Nx - 3, self.action_dict["up"]] = r_goal
        R[self.Ny - 5, self.Nx - 3, self.action_dict["down"]] = r_goal
        R[self.Ny - 2, self.Nx - 4, self.action_dict["right"]] = r_goal
        R[self.Ny - 2, self.Nx - 2, self.action_dict["left"]] = r_goal
        return R

    def _build_wind(self):
        wind = np.zeros(self.Nx, dtype = int)
        wind[3:6] = 1
        wind[6:8] = 2
        wind[8] = 1
        return wind

class Agent:

    def __init__(self, env):
        # Store state and action dimension
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        # Agent learning parameters
        self.epsilon = 1.000  # initial exploration probability
        self.epsilon_decay = 0.001  # epsilon decay after each episode
        self.beta = 0.999  # learning rate
        self.gamma = 0.999  # reward discount factor
        # Initialize Q[s,a] table
        self.Q = np.zeros(self.state_dim + self.action_dim, dtype=float)

    def get_action(self, env):
        # Epsilon-greedy agent policy
        if random.uniform(0, 1) < self.epsilon:
            # explore
            return np.random.choice(env.allowed_actions())
        else:
            # exploit on allowed actions
            state = env.state;
            actions_allowed = env.allowed_actions()
            Q_s = self.Q[state[0], state[1], actions_allowed]
            actions_greedy = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            return np.random.choice(actions_greedy)

    def train(self, memory):
        # -----------------------------
        # Update:
        #
        # Q[s,a] <- Q[s,a] + beta * (R[s,a] + gamma * max(Q[s,:]) - Q[s,a])
        #
        #  R[s,a] = reward for taking action a from state s
        #  beta = learning rate
        #  gamma = discount factor
        # -----------------------------
        (state, action, state_next, reward, done) = memory
        sa = state + (action,)
        self.Q[sa] += self.beta * (reward + self.gamma * np.max(self.Q[state_next]) - self.Q[sa])

    def display_greedy_policy(self):
        # greedy policy = argmax[a'] Q[s,a']
        greedy_policy = np.zeros((self.state_dim[0], self.state_dim[1]), dtype=int)
        for x in range(self.state_dim[1]):
            for y in range(self.state_dim[0]):
                greedy_policy[y, x] = np.argmax(self.Q[y, x, :])
        print("\nGreedy policy(y, x):")
        print(greedy_policy)
        print()

def main():

    # Settings
    env = Environment(Ny = 7, Nx = 10)
    agent = Agent(env)

    # Train agent
    print("\nTraining agent...\n")
    N_episodes = 1000
    for episode in range(N_episodes):

        # Generate an episode
        iter_episode, reward_episode = 0, 0
        state = env.reset()  # starting state
        while True:
            action = agent.get_action(env)  # get action
            state_next, reward, done = env.step(action)  # evolve state by action
            agent.train((state, action, state_next, reward, done))  # train agent
            iter_episode += 1
            reward_episode += reward
            if done:
                break
            state = state_next  # transition to next state

        # Decay agent exploration parameter
        agent.epsilon = max(agent.epsilon - agent.epsilon_decay, 0.01)

        # Print
        if (episode == 0) or (episode + 1) % 50 == 0:
            print("[episode {}/{}] eps = {:.3F} -> iter = {}, rew = {:.1F}".format(
                episode + 1, N_episodes, agent.epsilon, iter_episode, reward_episode))

        # Print greedy policy
        if (episode == N_episodes - 1):
            agent.display_greedy_policy()
            for (key, val) in sorted(env.action_dict.items(), key=operator.itemgetter(1)):
                print(" action['{}'] = {}".format(key, val))
            print()

# Driver
if __name__ == '__main__':
    main()