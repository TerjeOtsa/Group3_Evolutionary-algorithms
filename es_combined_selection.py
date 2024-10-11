import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Laster inn data
data_folder = './'  # Fra filer i samme mappe
monthly_returns_df = pd.read_csv(data_folder + 'Monthly_Returns_Data.csv', index_col=0)
covariance_matrix = pd.read_csv(data_folder + 'Covariance_Matrix.csv', index_col=0)

# Portfolio Evalutiation funksjon
def portfolio_return(weights, returns):
    return np.dot(weights, returns.mean())

def portfolio_risk(weights, covariance_matrix):
    return np.dot(weights.T, np.dot(covariance_matrix, weights))

# Initialiserer population,tilfeldige weights
def initialize_population(size, num_assets):
    population = []
    mutation_rates = []
    for _ in range(size):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)  # Sørger for at weights sum er 1
        population.append(weights)
        mutation_rates.append(0.05)  # Initaliserer mutation rates
    return np.array(population), np.array(mutation_rates)

# Fitness funksjon, forventet retur 
def fitness_function(weights, returns):
    return np.dot(weights, returns.mean())

# Mutation med selv-adaptive mutation rate
def mutate(weights, mutation_rate):
    mutation = np.random.normal(0, mutation_rate, len(weights))
    weights += mutation
    weights = np.clip(weights, 0, 1)
    weights /= np.sum(weights)
    return weights

# Avansert recombination funksjon
def recombine(parents):
    parent1, parent2 = parents
    child = (parent1 + parent2) / 2  
    return child / np.sum(child)

# Selection funksjon (fra begge parents og offspring)
def select_best_combined(population, fitness, num_selected):
    best_indices = np.argsort(fitness)[-num_selected:]
    return population[best_indices]

# (μ + λ) Evolutionary Strategies (ES)
def mu_plus_lambda_es(returns, covariance_matrix, population_size, offspring_size, num_generations, mutation_rate):
    num_assets = returns.shape[1]
    population, mutation_rates = initialize_population(population_size, num_assets)
    best_fitness_per_generation = []

    for generation in range(num_generations):
        # Step 1: Evaluate fitness of parents
        fitness = np.array([fitness_function(ind, returns) for ind in population])

        # Step 2: Generate offspring by recombining and mutating parents
        offspring = []
        for _ in range(offspring_size):
            parents = np.random.choice(len(population), size=2, replace=False)
            child = recombine(population[parents])
            child = mutate(child, mutation_rate)
            offspring.append(child)
        offspring = np.array(offspring)

        # Step 3: Evaluate fitness of offspring
        offspring_fitness = np.array([fitness_function(ind, returns) for ind in offspring])

        # Step 4: Combine parents and offspring populations
        combined_population = np.vstack((population, offspring))
        combined_fitness = np.hstack((fitness, offspring_fitness))

        # Step 5: Select the best individuals from both parents and offspring
        population = select_best_combined(combined_population, combined_fitness, population_size)

        # Track the best fitness in each generation
        best_fitness_per_generation.append(np.max(combined_fitness))

    best_portfolio = population[np.argmax(fitness)]
    return best_portfolio, best_fitness_per_generation

# Define the parameters for the (μ + λ) Evolutionary Strategies algorithm
population_size = 100  # μ: Number of parents in the population
offspring_size = 150   # λ: Number of offspring to generate
num_generations = 500  # Number of generations to run the algorithm
mutation_rate = 0.05  # Mutation rate to introduce variations

# Run (μ + λ) Evolutionary Strategies (ES)
best_portfolio_es, fitness_history_es = mu_plus_lambda_es(
    monthly_returns_df, covariance_matrix, population_size, offspring_size, num_generations, mutation_rate
)

# Evaluate the best ES portfolio
best_portfolio_return_es = portfolio_return(best_portfolio_es, monthly_returns_df)
best_portfolio_risk_es = portfolio_risk(best_portfolio_es, covariance_matrix)
best_portfolio_volatility_es = np.sqrt(best_portfolio_risk_es)

# Print the ES portfolio results
print("\nBest Portfolio Expected Return (μ + λ ES):", best_portfolio_return_es)
print("Best Portfolio Risk (Variance) (μ + λ ES):", best_portfolio_risk_es)
print("Best Portfolio Volatility (Standard Deviation) (μ + λ ES):", best_portfolio_volatility_es)

# Plot fitness history for (μ + λ) ES
plt.plot(fitness_history_es)
plt.xlabel('Generation')
plt.ylabel('Best Fitness (Expected Return)')
plt.title('Evolution of Portfolio Expected Return (μ + λ ES)')
plt.show()
