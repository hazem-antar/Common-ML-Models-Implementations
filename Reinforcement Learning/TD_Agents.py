import numpy as np
import time

class TD_Agent:
    '''
    Class used to train a temporal difference agent of one of three algorithm types:
    Sarsa / Expected Sarsa / Q-learning
    '''
    def __init__(self, env, method = 'q_learning', alpha = 0.7, gamma = 0.95, 
                 epsilon = 1.0, epsilon_decay = 0.995, epsilon_min = 0.1):
        
        # Initializing Q-table with zeros
        self.Q = np.zeros((env.observation_space_n, env.action_space_n)) 

        # Learning rate
        self.alpha = alpha  

        # Discount factor
        self.gamma = gamma 

        # Probability cofficinet to control amount of exploration/exploitation
        self.epsilon = epsilon 

        # Decay rate for epsilon (faster decay leads to less exploration)
        self.epsilon_decay = epsilon_decay  

        # Minimum value for epsilon
        self.epsilon_min = epsilon_min 

        # Number of actions in the action space 
        self.num_actions = env.action_space_n  

        # Type of TD algorithm used in training
        self.TD_method = method

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

    def update_policy(self, state, action, reward, next_state, next_action = None):
        '''
        Method to update the policy (Q-table) based on a specific TD method
        '''
        # Compute the next Q-value based on the chosen method
        if self.TD_method == 'sarsa':
            next_Q = self.Q[next_state][next_action]

        elif self.TD_method == 'expected_sarsa':
            # Calculate probability of each possible action given the epsilon-greedy strategy
            actions_probs = (self.epsilon / self.num_actions) + (1 - self.epsilon) * (self.Q[next_state] == np.max(self.Q[next_state]))
            # Calculate the expectation by weight each Q[S,A] by its probability
            next_Q = np.dot(actions_probs, self.Q[next_state])
        
        else:  # Q-learning
            next_Q = np.max(self.Q[next_state])

        # Update the Q-value 
        target = reward + self.gamma * next_Q
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])

    def decay_epsilon(self):
        '''
        Method used for a scheduled update to the epsilon decay rate
        '''
        # Decay epsilon until it reaches its minimum value
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, episodes, max_steps_per_episode, render = False, file_path = "q_learning_table.pkl"):
        '''
        Method for training the TD agent given a specific type of algorithm
        '''
        # Instantiate lists to log the training data
        episode_rewards = []
        episode_lengths = []
        episode_epsilons = []

        # Iterate over episodes
        for _ in range(episodes):

            # Reset the environment and observe initial state
            state = env.reset()

            # Choose an action using epsilon-greedy strategy
            action = self.eps_greedy_action(state)

            # Initialize log variables
            total_reward = 0
            step_count = 0

            # Iterate over the episode timesteps
            for _ in range(max_steps_per_episode):

                # Render the environment (for visualization but will slow down the training)
                if render:
                    env.render()
                    time.sleep(0.25)
                
                # Take a step and observe rewards and next state
                next_state, reward, done = env.step(action)
                
                # Initialize next action
                next_action = None

                # Choose next action before policy update (only if not training a Q-learning agent)
                if self.TD_method != 'q_learning':
                    next_action = self.eps_greedy_action(next_state) 
                
                # Update the agent's policy
                self.update_policy(state, action, reward, next_state, next_action)

                # Choose next action after policy update (only if training a Q-learning agent)
                if self.TD_method == 'q_learning':
                    next_action = self.eps_greedy_action(next_state) 

                # Update variables for next iteration
                state = next_state
                action = next_action

                # Update log variables
                total_reward += reward
                step_count += 1

                # If episode finished, break
                if done:
                    break
            
            # Append log variables to the lists
            episode_rewards.append(total_reward)
            episode_lengths.append(step_count)
            episode_epsilons.append(self.epsilon)

            # Decay the epsilon coefficient
            self.decay_epsilon()

        # At the end of training, dumb the Q-table
        self.save_q_table(file_path)

        return episode_rewards, episode_lengths, episode_epsilons

    def save_q_table(self, file_path):
        '''
        Method to dumb the Q-table
        '''
        # Save the Q-table to a file
        np.save(file_path, self.Q)

        print(f"Q-table saved to {file_path}")

