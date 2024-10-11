import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Portfolio evaluation functions
def portfolio_return(weights, returns):
    return np.dot(weights, returns.mean())

def portfolio_risk(weights, covariance_matrix):
    return np.dot(weights.T, np.dot(covariance_matrix, weights))

# Initialize population (random weights)
def initialize_population(size, num_assets):
    population = []
    mutation_rates = []
    for _ in range(size):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)  # Ensure weights sum to 1
        population.append(weights)
        mutation_rates.append(0.05)  # Initialize mutation rates
    return np.array(population), np.array(mutation_rates)

# Fitness function (expected return)
def fitness_function(weights, returns):
    return np.dot(weights, returns.mean())

# Mutation with self-adaptive mutation rate
def mutate(weights, mutation_rate):
    mutation = np.random.normal(0, mutation_rate, len(weights))
    weights += mutation
    weights = np.clip(weights, 0, 1)
    weights /= np.sum(weights)
    return weights

# Advanced recombination function
def recombine(parents):
    parent1, parent2 = parents
    child = (parent1 + parent2) / 2  # Intermediary recombination
    return child / np.sum(child)

# Evolutionary Strategies (μ, λ) with self-adaptive mutation rate
def offspring_selection_es(returns, population_size, offspring_size, num_generations, mutation_rate):
    num_assets = returns.shape[1]
    population, mutation_rates = initialize_population(population_size, num_assets)
    best_fitness_per_generation = []

    for generation in range(num_generations):
        # Step 1: Evaluate fitness
        fitness = np.array([fitness_function(ind, returns) for ind in population])

        # Step 2: Recombination (generate offspring by recombining parents)
        offspring = []
        for _ in range(offspring_size):
            parents = np.random.choice(len(population), size=2, replace=False)
            child = recombine(population[parents])
            offspring.append(child)

        offspring = np.array(offspring)

        # Step 3: Mutation (apply small mutations to offspring)
        offspring = np.array([mutate(child, mutation_rate) for child in offspring])

        # Step 4: Evaluate offspring fitness
        offspring_fitness = np.array([fitness_function(ind, returns) for ind in offspring])

        # Step 5: Select only from offspring
        best_indices = np.argsort(offspring_fitness)[-population_size:]  # Select best from offspring only
        population = offspring[best_indices]
        fitness = offspring_fitness[best_indices]

        # Track the best fitness in each generation
        best_fitness_per_generation.append(np.max(fitness))

    best_portfolio = population[np.argmax(fitness)]
    return best_portfolio, best_fitness_per_generation

# Define the parameters for the Evolutionary Strategies algorithm
population_size = 150  # Number of portfolios in the population (μ)
offspring_size = 300  # Number of offspring (λ)
num_generations = 500  # Number of generations to run the algorithm
mutation_rate = 0.05  # Mutation rate to introduce variations

# Load your data
data_folder = './'  # Assuming the files are in the same directory
monthly_returns_df = pd.read_csv(data_folder + 'Monthly_Returns_Data.csv', index_col=0)
covariance_matrix = pd.read_csv(data_folder + 'Covariance_Matrix.csv', index_col=0)

# Run the (μ, λ) Evolutionary Strategies
best_portfolio_es, fitness_history_es = offspring_selection_es(
    monthly_returns_df, population_size, offspring_size, num_generations, mutation_rate
)

# Evaluate the best ES portfolio
best_portfolio_return_es = portfolio_return(best_portfolio_es, monthly_returns_df)
best_portfolio_risk_es = portfolio_risk(best_portfolio_es, covariance_matrix)
best_portfolio_volatility_es = np.sqrt(best_portfolio_risk_es)

# Print the ES portfolio results
print("\nBest Portfolio Expected Return (μ, λ ES):", best_portfolio_return_es)
print("Best Portfolio Risk (Variance) (μ, λ ES):", best_portfolio_risk_es)
print("Best Portfolio Volatility (Standard Deviation) (μ, λ ES):", best_portfolio_volatility_es)

# Plot fitness history for (μ, λ) ES
plt.plot(fitness_history_es)
plt.xlabel('Generation')
plt.ylabel('Best Fitness (Expected Return)')
plt.title('Evolution of Portfolio Expected Return (μ, λ ES)')
plt.show()
