import numpy as np
import random


class DynaQ_Agent:
    '''
    Class to instantiate a Dyna-Q model
    '''
    def __init__(self, env, alpha, gamma, epsilon, planning_steps):
        # Set object parameters
        self.env = env
        self.num_actions = env.action_space_n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Set the number of planning steps
        self.planning_steps = planning_steps
        # Initialize the Q-table with zeros
        self.Q = np.zeros((env.observation_space_n, env.action_space_n))
        # Model of environment
        self.model = {}

    def eps_greedy_action(self, state):
        '''
        Method used to select action based on epsilon-greedy strategy
        '''
        if np.random.rand() < self.epsilon:
            # Choose random action (including the greedy)
            return np.random.choice(self.num_actions)
        else:
            # Choose greedy action 
            return np.argmax(self.Q[state])

    def update_model(self, state, action, reward, next_state):
        '''
        Method to update the internal model with observed next state and reward
        '''
        # If the state is not yet in the model, add it
        if state not in self.model:
            self.model[state] = {}
        # Store the observed transition and reward
        self.model[state][action] = (next_state, reward)

    def simulate_experience(self):
        '''
        Method to simulate experiences based on the learned model to update Q-values
        '''
        # Repeat for a number of planning steps
        for _ in range(self.planning_steps):
            
            # Only continue if the model has been populated
            # Because it's a deterministic environment so nothing will change
            if not self.model:
                continue

            # Randomly pick a pre-visited state from the learned model dictionary
            s = random.choice(list(self.model.keys()))

            # Randomly pick an action for this state
            a = random.choice(list(self.model[s].keys()))

            # Get the next state and reward from the model
            next_s, r = self.model[s][a]

            # Find the best action at the next state
            best_next_a = np.argmax(self.Q[next_s])

            # Update the Q-value using the simulated experience
            self.Q[s, a] += self.alpha * (r + self.gamma * self.Q[next_s, best_next_a] - self.Q[s, a])

    def train(self, total_episodes, file_path, epsilon_decay=0.995, epsilon_min=0.1):
        '''
        Method to train the agent over a specified number of episodes
        '''

        # Initialize the log lists
        episode_rewards = []
        episode_lengths = []
        episode_epsilons = []

        # Loop over each episode
        for _ in range(total_episodes):

            # Reset the environment for the new episode
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            # Continue until episode is finished
            while not done:

                # Choose an action based on eps_greedy strategy
                action = self.eps_greedy_action(state)

                # Take a step in real environment and observe the results
                next_state, reward, done = self.env.step(action)

                # Update the model with the observed next state and reward
                self.update_model(state, action, reward, next_state)

                # Update the Q-values with the observed next state and reward
                self.direct_learn(state, action, reward, next_state)

                # Simulate additional experiences
                self.simulate_experience()

                # Move to the next state
                state = next_state

                # Accumulate the reward
                total_reward += reward

                # Increment the step counter
                steps += 1
            
            # Log the results of the episode
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            episode_epsilons.append(self.epsilon)

            # Apply epsilon decay after the episode
            self.epsilon = max(epsilon_min, self.epsilon * epsilon_decay)

        # Save the learned Q-table
        self.save_q_table(file_path)

        return episode_rewards, episode_lengths, episode_epsilons

    
    def direct_learn(self, state, action, reward, next_state):
        '''
        Method to directly learn from real experience
        '''
        # Find the best possible action at the next state
        best_next_action = np.argmax(self.Q[next_state])
        # Update the Q-value for the current state and action
        self.Q[state, action] += self.alpha * (reward + self.gamma * self.Q[next_state, best_next_action] - self.Q[state, action])

    def save_q_table(self, file_path):
        '''
        Method to dumb the Q-table
        '''
        # Save the Q-table to a file
        np.save(file_path, self.Q)

        print(f"Q-table saved to {file_path}")
