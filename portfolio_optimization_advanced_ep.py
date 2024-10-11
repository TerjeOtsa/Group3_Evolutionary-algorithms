import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from your CSV files
data_folder = './'  # Assuming the files are in the same directory

monthly_returns_df = pd.read_csv(data_folder + 'Monthly_Returns_Data.csv', index_col=0)
covariance_matrix = pd.read_csv(data_folder + 'Covariance_Matrix.csv', index_col=0)

# Proceed with your Evolutionary Programming code
num_stocks = len(monthly_returns_df.columns)  # Number of stocks
population_size = 150  # Number of portfolios in the population
num_generations = 500  # Number of iterations
initial_mutation_rate = 0.1  # Increase initial mutation rate for more exploration
elitism_size = 3  # Reduce elitism to preserve fewer individuals

# Initialize population (random weights)
def initialize_population(size, num_assets):
    population = []
    mutation_rates = []  # Initialize mutation rates for each portfolio
    for _ in range(size):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)  # Ensure weights sum to 1
        population.append(weights)
        mutation_rates.append(initial_mutation_rate)  # Start with the same mutation rate for all
    return np.array(population), np.array(mutation_rates)

# Fitness function (expected return)
def fitness_function(weights, returns):
    return np.dot(weights, returns.mean())

# Mutation with self-adaptive strategy
def mutate(weights, mutation_rate):
    mutation = np.random.normal(0, mutation_rate, len(weights))  # Small random change
    weights += mutation
    weights = np.clip(weights, 0, 1)  # Ensure weights are in the range [0, 1]
    weights /= np.sum(weights)  # Ensure weights sum to 1
    return weights

# Define the portfolio evaluation functions
def portfolio_return(weights, returns):
    return np.dot(weights, returns.mean())

def portfolio_risk(weights, covariance_matrix):
    return np.dot(weights.T, np.dot(covariance_matrix, weights))

# Evolutionary Programming with Elitism and Self-Adaptive Mutation
def advanced_evolutionary_programming(returns, population_size, num_generations, elitism_size):
    population, mutation_rates = initialize_population(population_size, returns.shape[1])
    best_fitness_per_generation = []  # To track the best fitness in each generation

    for generation in range(num_generations):
        # Calculate fitness for the current population
        fitness = np.array([fitness_function(ind, returns) for ind in population])

        # Sort by fitness to apply elitism
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        mutation_rates = mutation_rates[sorted_indices]
        fitness = fitness[sorted_indices]

        # Preserve the best individuals (elitism)
        elites = population[-elitism_size:]

        # Generate offspring through mutation with self-adaptive mutation rates
        new_population = []
        new_mutation_rates = []
        for i in range(population_size - elitism_size):
            parent = population[i]
            mutation_rate = mutation_rates[i]

            # Self-adaptive mutation: Adjust mutation rate dynamically
            if i < population_size // 2:
                mutation_rate *= 0.95  # Slow down mutation rate for top half of population
            else:
                mutation_rate *= 1.2  # Increase mutation rate for bottom half of population

            offspring = mutate(parent, mutation_rate)
            new_population.append(offspring)
            new_mutation_rates.append(mutation_rate)

        # Combine elites and offspring to form the new population
        population = np.vstack((new_population, elites))
        mutation_rates = np.array(new_mutation_rates + [initial_mutation_rate] * elitism_size)

        # Track the best fitness in each generation
        best_fitness_per_generation.append(np.max(fitness))

    # Return the best portfolio and fitness history
    return population[np.argmax(fitness)], best_fitness_per_generation

# Running the advanced EP algorithm
best_portfolio_weights, fitness_history = advanced_evolutionary_programming(monthly_returns_df, population_size, num_generations, elitism_size)

# Evaluate the best portfolio found by the algorithm
best_portfolio_return = portfolio_return(best_portfolio_weights, monthly_returns_df)
best_portfolio_risk = portfolio_risk(best_portfolio_weights, covariance_matrix)
best_portfolio_volatility = np.sqrt(best_portfolio_risk)

# Print the portfolio performance
print("Best Portfolio Expected Return:", best_portfolio_return)
print("Best Portfolio Risk (Variance):", best_portfolio_risk)
print("Best Portfolio Volatility (Standard Deviation):", best_portfolio_volatility)

# Plot fitness evolution over generations
plt.plot(fitness_history)
plt.xlabel('Generation')
plt.ylabel('Best Fitness (Expected Return)')
plt.title('Evolution of Portfolio Expected Return (Advanced EP)')
plt.show()

# Display the best portfolio weights
print(best_portfolio_weights)