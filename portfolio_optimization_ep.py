import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from your CSV files
data_folder = './'  # Update with your path

monthly_returns_df = pd.read_csv(data_folder + 'Monthly_Returns_Data.csv', index_col=0)
covariance_matrix = pd.read_csv(data_folder + 'Covariance_Matrix.csv', index_col=0)

# Proceed with your Evolutionary Programming code
num_stocks = len(monthly_returns_df.columns)  # Number of stocks
population_size = 100  # Number of portfolios in the population
num_generations = 500  # Number of iterations
mutation_rate = 0.01  # Probability of mutation

# Initialize population (random weights)
def initialize_population(size, num_assets):
    population = []
    for _ in range(size):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)  # Ensure weights sum to 1
        population.append(weights)
    return np.array(population)

# Fitness function (expected return)
def fitness_function(weights, returns):
    return np.dot(weights, returns.mean())


# Mutation
def mutate(weights, rate):
    mutation = np.random.normal(0, rate, len(weights))  # Small random change
    weights += mutation
    weights = np.clip(weights, 0, 1)  # Ensure weights are in the range [0, 1]
    weights /= np.sum(weights)  # Ensure weights sum to 1
    return weights

# Define the portfolio evaluation functions
def portfolio_return(weights, returns):
    return np.dot(weights, returns.mean())

def portfolio_risk(weights, covariance_matrix):
    return np.dot(weights.T, np.dot(covariance_matrix, weights))




def evolutionary_programming(returns, population_size, num_generations, mutation_rate):
    population = initialize_population(population_size, returns.shape[1])
    best_fitness_per_generation = []  # To track the best fitness in each generation
    for generation in range(num_generations):
        fitness = np.array([fitness_function(ind, returns) for ind in population])
        offspring = np.array([mutate(ind, mutation_rate) for ind in population])
        offspring_fitness = np.array([fitness_function(ind, returns) for ind in offspring])
        combined_population = np.vstack((population, offspring))
        combined_fitness = np.hstack((fitness, offspring_fitness))
        best_indices = np.argsort(combined_fitness)[-population_size:]  # Select best solutions
        population = combined_population[best_indices]
        
        # Track the best fitness in each generation
        best_fitness_per_generation.append(np.max(fitness))
    
    return population[np.argmax(fitness)], best_fitness_per_generation

# Running the updated EP algorithm
best_portfolio_weights, fitness_history = evolutionary_programming(monthly_returns_df, population_size, num_generations, mutation_rate)

# Evaluate the best portfolio found by the algorithm
best_portfolio_return = portfolio_return(best_portfolio_weights, monthly_returns_df)
best_portfolio_risk = portfolio_risk(best_portfolio_weights, covariance_matrix)
best_portfolio_volatility = np.sqrt(best_portfolio_risk)

# Print the portfolio performance
print("Best Portfolio Expected Return:", best_portfolio_return)
print("Best Portfolio Risk (Variance):", best_portfolio_risk)
print("Best Portfolio Volatility (Standard Deviation):", best_portfolio_volatility)


plt.plot(fitness_history)
plt.xlabel('Generation')
plt.ylabel('Best Fitness (Expected Return)')
plt.title('Evolution of Portfolio Expected Return')
plt.show()

# Display the best portfolio weights
print(best_portfolio_weights)
