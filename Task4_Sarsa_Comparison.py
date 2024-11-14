import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the Taxi environment
env = gym.make("Taxi-v3", render_mode="ansi")

# Hyperparameters
alpha = 0.1           # Learning rate
gamma = 0.8           # Discount factor
epsilon = 1.0         # Initial exploration rate
epsilon_decay = 0.995 # Decay rate for epsilon
min_epsilon = 0.01    # Minimum exploration rate
num_episodes = 20000  # Maximum episodes for training
evaluation_interval = 1000  # Evaluate every 1000 episodes
final_evaluation_episodes = 100  # Number of episodes for final evaluation
evaluation_episodes = 10  # Number of episodes for intermediate evaluations

# Function to evaluate the policy with a given Q-table
def evaluate_policy(q_table, num_episodes=10, max_steps=200):
    """
    Evaluates the given Q-table by running a set number of episodes.
    
    Parameters:
    - q_table: The Q-table to evaluate.
    - num_episodes: Number of episodes to run for evaluation.
    - max_steps: Maximum steps per episode.
    
    Returns:
    - The average reward over the evaluation episodes.
    """
    total_rewards = []
    for episode in range(num_episodes):
        state, done = env.reset()[0], False
        episode_reward = 0
        for _ in range(max_steps):
            action = np.argmax(q_table[state])
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            if done:
                break
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

# Function to run the Q-learning algorithm with periodic evaluations
def q_learning(env):
    """
    Implements the Q-learning algorithm and performs periodic evaluations.
    
    Parameters:
    - env: The Gym environment.
    
    Returns:
    - rewards: List of rewards per episode.
    - eps_values: List of epsilon values over episodes.
    - q_table: The final Q-table after training.
    - evaluation_results: Evaluation results at specified intervals.
    """
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards = []
    eps_values = []
    evaluation_results = []
    eps = epsilon
    
    for episode in range(num_episodes + 1):
        state, done = env.reset()[0], False
        total_reward = 0
        
        # Episode loop: train the agent until episode is complete
        while not done:
            # Epsilon-greedy action selection
            action = env.action_space.sample() if np.random.rand() < eps else np.argmax(q_table[state])
            next_state, reward, done, truncated, _ = env.step(action)
            # Q-learning update rule
            q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
            total_reward += reward
            state = next_state
        
        rewards.append(total_reward)
        eps_values.append(eps)
        eps = max(min_epsilon, eps * epsilon_decay)  # Decay epsilon

        # Evaluate the policy every 1000 episodes
        if episode % evaluation_interval == 0 and episode > 0:
            avg_reward = evaluate_policy(q_table, evaluation_episodes)
            evaluation_results.append((episode, avg_reward))
            print(f"Q-learning Evaluation at Episode {episode}: Average Reward = {avg_reward}")
        
    # Final evaluation with 100 episodes
    final_avg_reward = evaluate_policy(q_table, final_evaluation_episodes)
    evaluation_results.append((num_episodes, final_avg_reward))
    print(f"Q-learning Final Evaluation: Average Reward over {final_evaluation_episodes} episodes = {final_avg_reward}")
    
    return rewards, eps_values, q_table, evaluation_results

# Function to run the SARSA algorithm with periodic evaluations
def sarsa(env):
    """
    Implements the SARSA algorithm and performs periodic evaluations.
    
    Parameters:
    - env: The Gym environment.
    
    Returns:
    - rewards: List of rewards per episode.
    - eps_values: List of epsilon values over episodes.
    - q_table: The final Q-table after training.
    - evaluation_results: Evaluation results at specified intervals.
    """
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards = []
    eps_values = []
    evaluation_results = []
    eps = epsilon
    
    for episode in range(num_episodes + 1):
        state, done = env.reset()[0], False
        total_reward = 0
        # Initial action selection for SARSA
        action = env.action_space.sample() if np.random.rand() < eps else np.argmax(q_table[state])
        
        # Episode loop: train the agent until episode is complete
        while not done:
            next_state, reward, done, truncated, _ = env.step(action)
            # Epsilon-greedy selection for the next action
            next_action = env.action_space.sample() if np.random.rand() < eps else np.argmax(q_table[next_state])
            # SARSA update rule
            q_table[state, action] += alpha * (reward + gamma * q_table[next_state, next_action] - q_table[state, action])
            total_reward += reward
            state, action = next_state, next_action
        
        rewards.append(total_reward)
        eps_values.append(eps)
        eps = max(min_epsilon, eps * epsilon_decay)  # Decay epsilon

        # Evaluate the policy every 1000 episodes
        if episode % evaluation_interval == 0 and episode > 0:
            avg_reward = evaluate_policy(q_table, evaluation_episodes)
            evaluation_results.append((episode, avg_reward))
            print(f"SARSA Evaluation at Episode {episode}: Average Reward = {avg_reward}")
        
    # Final evaluation with 100 episodes
    final_avg_reward = evaluate_policy(q_table, final_evaluation_episodes)
    evaluation_results.append((num_episodes, final_avg_reward))
    print(f"SARSA Final Evaluation: Average Reward over {final_evaluation_episodes} episodes = {final_avg_reward}")
    
    return rewards, eps_values, q_table, evaluation_results

# Run both algorithms
print("Running Q-learning...")
q_rewards, q_eps_values, q_q_table, q_eval_results = q_learning(env)
print("Running SARSA...")
sarsa_rewards, sarsa_eps_values, sarsa_q_table, sarsa_eval_results = sarsa(env)

# Plot evaluation results for comparison
sns.set(style="whitegrid")

# Extract episode and reward data for plotting
plt.figure(figsize=(12, 5))
q_eval_x, q_eval_y = zip(*q_eval_results)
sarsa_eval_x, sarsa_eval_y = zip(*sarsa_eval_results)
plt.plot(q_eval_x, q_eval_y, label="Q-learning", marker='o')
plt.plot(sarsa_eval_x, sarsa_eval_y, label="SARSA", marker='o')
plt.xlabel("Training Episodes")
plt.ylabel("Average Reward (Evaluation)")
plt.title("Evaluation Results Every 1000 Episodes and Final 100-Episode Evaluation")
plt.legend()
plt.show()

# Cleanup environment
env.close()

print("Evaluation complete.")
