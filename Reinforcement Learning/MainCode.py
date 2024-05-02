from TD_Agents import *
from DynaQ import *
from Value_Iteration import *
from  Helper_Functions import *
from  Environment import *
import numpy as np

# Seed the pseudo number generator (My ID ends with 0393)
np.random.seed(3093)

# 1) Applying Value Iteration algorithm to get optimal policy and state-value function: -----------------

# Instantiate the environment
env = MazeEnv()  

# Instantiate a Value Iteration Agent
agent = ValueIteration_Agent(env)

# Train the Agent and get the optimal functions
V_optimal, policy_optimal = agent.train()

# Visualizing state-value function and action-value function on the maze
visualizer = MazeVisualizer(V = V_optimal, policy = policy_optimal).start(title = "ValueIteration")


# 2) Train and test TD agent of types (Sarsa / Expected Sarsa / Q-learning): -----------------
for TD in  ['sarsa', 'expected_sarsa', 'q_learning']:
    
    # Instantiate the environment
    env = MazeEnv()

    # Instantiate a TD Agent from a specific type
    td_Agent = TD_Agent(env, method=TD)

    # Train the agent, log training data, and dump the final Q-table
    train_rewards, train_lengths, train_epsilons = td_Agent.train(env, episodes = 10000, max_steps_per_episode = 3000,
                                                                render = False, file_path = "TD_q_table.npy")

    # Test the agent and log the testing data
    test_rewards, test_lengths = test_agent(td_Agent, env, episodes = 1000, max_steps_per_episode = 1000, render = False)

    # Visualize the logged data (Train vs Test)
    plot_logs(train_rewards, train_lengths, train_epsilons, test_rewards, test_lengths, window_size = 50)

    # Extract state-value function and policy
    V, policy = Extract_V_PI('TD_q_table.npy')

    # Compare to baseline 
    compare_algorithms(env, V = V, policy = policy, V_optimal = V_optimal,
                        policy_optimal = policy_optimal, method_name = TD)
    
    # Close the environment
    pygame.quit()

    # Visualizing state-value function and action-value function on the maze
    visualizer = MazeVisualizer(V = V, policy = policy).start(title = TD)


# 3) Train and test Dyna-Q agent: -----------------

# Instantiate the environment
env = MazeEnv()

# Instantiate DynaQ Agent
DQ_agent = DynaQ_Agent(env, alpha = 0.1, gamma = 0.95, epsilon = 1, planning_steps=5)

train_rewards, train_lengths, train_epsilons = DQ_agent.train(total_episodes=10000, file_path = 'Dyna_Q_table.npy')

 # Test the agent and log the testing data
test_rewards, test_lengths = test_agent(DQ_agent, env, episodes = 1000, max_steps_per_episode = 1000, render = False)

# Visualize the logged data (Train vs Test)
plot_logs(train_rewards, train_lengths, train_epsilons, test_rewards, test_lengths, window_size = 50)

# Extract state-value function and policy
V, policy = Extract_V_PI('Dyna_Q_table.npy')

# Compare to optimal 
compare_algorithms(env, V = V, policy = policy, V_optimal = V_optimal,
                    policy_optimal = policy_optimal, method_name = "Dyna_Q")

# Close the environment
pygame.quit()

# Visualizing state-value function and action-value function on the maze
visualizer = MazeVisualizer(V = V, policy = policy).start(title = "Dyna_Q")
