import numpy as np

class ValueIteration_Agent:
    '''
    Class to train a Value Iteration agent
    '''
    def __init__(self, env, gamma=0.95, theta=1e-10):

        # Set the object parameters
        self.env = env  
        self.gamma = gamma  
        
        # Convergence threshold for stopping the iteration
        self.theta = theta 

        # Initializing Value function
        self.V = np.zeros(env.observation_space_n) 

        # Initializing Policy function
        self.policy = np.zeros(env.observation_space_n, dtype=int)  

    def train(self):
        '''
        Method to training a value iteration agent
        '''
        # Initialize delta
        delta = np.inf  

        # Iterating until delta is < the threshold
        while delta > self.theta:
            
            # Reset delta to 0 at the beginning of iteration
            delta = 0 

            # Loop through all states
            for s in range(self.env.observation_space_n): 
                
                # Store the current V(s)
                v = self.V[s]  

                # Initialize the new V'(s)
                V_s = float('-inf')  

                # Loop through all possible actions
                for a in range(self.env.action_space_n):  

                    # Change the agent position to simulate taking action (a) at state (s)
                    self.env.agent_pos = (s // len(self.env.maze[0]), s % len(self.env.maze[0]))

                    # Take the action (a) and observe the reward and next state
                    next_state, reward, done = self.env.step(a)

                    # Calculate the Q-value for this state-action pair for only
                    # The observed reward and next state, because the P(s'|s, a) is zero
                    # For all other S' and the observed reward is also determinstic
                    Q_s_a = reward + (0 if done else self.gamma * self.V[next_state])

                    # Update maximum found V new so far for this state
                    V_s = max(V_s, Q_s_a)

                # Update the value function for this state with the maximum found
                self.V[s] = V_s

                # Update delta with the maximum change in value function (across all states)
                delta = max(delta, abs(v - self.V[s]))

        # After V converges, determine the optimal policy -------

        # Loop through all states
        for s in range(self.env.observation_space_n):

            # Initialize the best Q(s,a)
            best_Q_s_a = float('-inf')  

            # Initialize the best action
            best_action = None  

            # Loop through all possible actions
            for a in range(self.env.action_space_n):  
                
                # Change the agent position to simulate taking action (a) at state (s)
                self.env.agent_pos = (s // len(self.env.maze[0]), s % len(self.env.maze[0]))

                # Take the action (a) and observe the reward and next state
                next_state, reward, done = self.env.step(a)
                # Restore the original position

                # Compute Q(s,a)
                Q_s_a = reward + (0 if done else self.gamma * self.V[next_state])

                # If Q(s,a) is the best found so far, update the best Q(s,a) and best action
                if Q_s_a > best_Q_s_a:
                    best_Q_s_a = Q_s_a
                    best_action = a

            # Set the best action found as policy(s)
            self.policy[s] = best_action

        # Return the final value function and the optimal policy
        return self.V, self.policy

