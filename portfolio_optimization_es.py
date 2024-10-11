import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your data
data_folder = './'
monthly_returns_df = pd.read_csv(data_folder + 'Monthly_Returns_Data.csv', index_col=0)
covariance_matrix = pd.read_csv(data_folder + 'Covariance_Matrix.csv', index_col=0)

# Portfolio evaluation functions
def portfolio_return(weights, returns):
    return np.dot(weights, returns.mean())

def portfolio_risk(weights, covariance_matrix):
    return np.dot(weights.T, np.dot(covariance_matrix, weights))

# Initialize population (random weights)
def initialize_population(size, num_assets):
    population = []
    for _ in range(size):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        population.append(weights)
    return np.array(population)

# Fitness function (expected return)
def fitness_function(weights, returns):
    return np.dot(weights, returns.mean())

# Mutation function
def mutate(weights, rate):
    mutation = np.random.normal(0, rate, len(weights))
    weights += mutation
    weights = np.clip(weights, 0, 1)
    weights /= np.sum(weights)
    return weights

# Recombination function for ES
def recombine(parents):
    parent1, parent2 = parents
    child = (parent1 + parent2) / 2
    return child / np.sum(child)

# Selection function
def select_best(population, fitness, num_survivors):
    best_indices = np.argsort(fitness)[-num_survivors:]
    return population[best_indices]

# Evolutionary Strategies (ES) algorithm
def evolutionary_strategies(returns, population_size, num_generations, mutation_rate):
    num_assets = returns.shape[1]
    population = initialize_population(population_size, num_assets)
    best_fitness_per_generation = []

    for generation in range(num_generations):
        # Step 1: Evaluate fitness
        fitness = np.array([fitness_function(ind, returns) for ind in population])

        # Step 2: Selection (keep the best half of the population)
        num_survivors = population_size // 2
        population = select_best(population, fitness, num_survivors)

        # Step 3: Recombination (generate offspring by recombining parents)
        offspring = []
        for _ in range(population_size - num_survivors):
            parents = np.random.choice(len(population), size=2, replace=False)
            child = recombine(population[parents])
            offspring.append(child)

        offspring = np.array(offspring)

        # Step 4: Mutation (apply small mutations to offspring)
        offspring = np.array([mutate(child, mutation_rate) for child in offspring])

        # Step 5: Form the new population
        population = np.vstack((population, offspring))

        # Track the best fitness in each generation
        best_fitness_per_generation.append(np.max(fitness))

    best_portfolio = population[np.argmax(fitness)]
    return best_portfolio, best_fitness_per_generation

# Define the parameters for the Evolutionary Strategies algorithm
population_size = 100  # Number of portfolios in the population
num_generations = 500  # Number of generations to run the algorithm
mutation_rate = 0.01  # Mutation rate to introduce variations


# Run Evolutionary Strategies (ES)
best_portfolio_es, fitness_history_es = evolutionary_strategies(
    monthly_returns_df, population_size, num_generations, mutation_rate
)



# Evaluate the best ES portfolio
best_portfolio_return_es = portfolio_return(best_portfolio_es, monthly_returns_df)
best_portfolio_risk_es = portfolio_risk(best_portfolio_es, covariance_matrix)
best_portfolio_volatility_es = np.sqrt(best_portfolio_risk_es)

# Print the ES portfolio results
print("\nBest Portfolio Expected Return (ES):", best_portfolio_return_es)
print("Best Portfolio Risk (Variance) (ES):", best_portfolio_risk_es)
print("Best Portfolio Volatility (Standard Deviation) (ES):", best_portfolio_volatility_es)

# Plot fitness history for ES
plt.plot(fitness_history_es)
plt.xlabel('Generation')
plt.ylabel('Best Fitness (Expected Return)')
plt.title('Evolution of Portfolio Expected Return (ES)')
plt.show()


