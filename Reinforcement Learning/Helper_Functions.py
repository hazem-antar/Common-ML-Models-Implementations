import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time

def test_agent(agent, env, episodes , max_steps_per_episode, render=False):
    '''
    Method to test the trained agent
    '''
    # Instantiate lists to log the training data
    episode_rewards = []
    episode_lengths = []

    # Iterate over episodes
    for _ in range(episodes):
        
        # Reset the environment and observe initial state
        state = env.reset()

        # Choose an action from the optimal policy
        action = np.argmax(agent.Q[state])

        # Initialize log variables
        total_reward = 0
        step_count = 0

        # Iterate over the episode timesteps
        for _ in range(max_steps_per_episode):

            # Render the environment (for visualization)
            if render:
                env.render()
                time.sleep(0.25)
            
            # Take a step and observe rewards and next state
            state, reward, done = env.step(action)
            
            # Choose an action from the optimal policy
            action = np.argmax(agent.Q[state])
            
            # Update log variables
            total_reward += reward
            step_count += 1

            # If episode finished, break
            if done:
                break

        # Append log variables to the lists
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)

    return episode_rewards, episode_lengths

def plot_logs(train_rewards, train_lengths, train_epsilons, test_rewards, test_lengths, window_size = 50):
    '''
    Function to visualize the logged data (Train vs Test)
    '''
    # Create a window for a rolling average
    window = np.ones(window_size) / window_size
    
    # Compute rolling averages
    train_rewards_smoothed = np.convolve(train_rewards, window, mode='valid')
    train_lengths_smoothed = np.convolve(train_lengths, window, mode='valid')
    train_epsilons_smoothed = np.convolve(train_epsilons, window, mode='valid')
    train_epsilons_smoothed = np.convolve(train_epsilons, window, mode='valid')

    print("Last training ep. cumulative rewards:", train_rewards_smoothed[-1])
    print("Last training ep. lengths:", train_lengths_smoothed[-1])

    print("Testing mean ep. cumulative rewards:", np.mean(test_rewards))
    print("Testing mean ep. lengths:", np.mean(test_lengths))

    # Create a 3x1 subplot
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    
    # Plotting rewards
    axs[0].plot(train_rewards_smoothed, label='Trian - Episode Rewards (Moving Avg)')
    axs[0].axhline(y=np.mean(test_rewards), color='red', label='Test - Mean Episode Rewards')
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Total Reward')
    axs[0].set_title('Rewards per Episode')
    axs[0].legend()
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plotting episode lengths
    axs[1].plot(train_lengths_smoothed, label='Trian - Episode Lengths (Moving Avg)')
    axs[1].axhline(y=np.mean(test_lengths), color='red', label='Test - Mean Episode Lengths')
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Steps')
    axs[1].set_title('Episode Lengths')
    axs[1].legend()
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plotting episode epsilons
    axs[2].plot(train_epsilons_smoothed, label='Episode Epsilons (Moving Avg)')
    axs[2].set_xlabel('Episodes')
    axs[2].set_ylabel('Epsilon')
    axs[2].set_title('Episode Epsilons')
    axs[2].legend()
    axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def Extract_V_PI(filepath):
    '''
    Function to extract the policy and state-value functions from a Q-table
    '''
    # Load the Q-table from a file
    Q = np.load(filepath, allow_pickle=True)

    # Initialize value function
    V = np.zeros(len(Q))  

    # Initialize policy function
    policy = np.zeros(len(Q), dtype=int)  
    
    # Fill the functions
    for s in range(len(Q)):
        V[s] = np.max(Q[s])
        policy[s] = np.argmax(Q[s])
    
    return V, policy

def compare_algorithms(env, V, policy, V_optimal, policy_optimal, method_name):
    '''
    Function to compare the policy and state-value functions of two algorithms
    '''
    # Initialize variables
    hamming_distance = 0
    total = 0
    count = 0

    # Iterate over the maze cells
    for y in range(len(env.maze)):
        for x in range(len(env.maze[0])):

             # Check if cell is not a wall or terminal
            if env.maze[y][x] == 0:

                # Convert maze location to 1D state
                s = y * 10 + x

                # Calculate Hamming distance between policy and optimal policy
                hamming_distance += (policy_optimal[s] != policy[s])

                # Calculate MSE between value estimates and the optimal values 
                total += (V_optimal[s] - V[s]) ** 2
                count += 1

    print(f"Comparison Results for {method_name}:")
    print(f"\tHamming Distance (policy difference): {hamming_distance}")
    print(f"\tMSE (value function difference): {total/count:.4f}")