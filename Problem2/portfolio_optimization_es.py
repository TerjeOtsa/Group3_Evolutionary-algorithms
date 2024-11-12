import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_folder = 'Problem2/'
monthly_returns_df = pd.read_csv(data_folder + 'Monthly_Returns_Data.csv', index_col=0)
covariance_matrix = pd.read_csv(data_folder + 'Covariance_Matrix.csv', index_col=0)

# Portfolio evaluerings funksjoner
def portfolio_return(weights, returns):
    return np.dot(weights, returns.mean())

def portfolio_risk(weights, covariance_matrix):
    return np.dot(weights.T, np.dot(covariance_matrix, weights))

# Initialiserer population,tilfeldige weights
def initialize_population(size, num_assets):
    population = []
    for _ in range(size):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        population.append(weights)
    return np.array(population)

# Fitness funksjon (forventet retur)
def fitness_function(weights, returns):
    return np.dot(weights, returns.mean())

# Mutasjons funksjon
def mutate(weights, rate):
    mutation = np.random.normal(0, rate, len(weights))
    weights += mutation
    weights = np.clip(weights, 0, 1)
    weights /= np.sum(weights)
    return weights

# rekombinasjons funksjon for ES
def recombine(parents):
    parent1, parent2 = parents
    child = (parent1 + parent2) / 2
    return child / np.sum(child)

# Selection funksjon
def select_best(population, fitness, num_survivors):
    best_indices = np.argsort(fitness)[-num_survivors:]
    return population[best_indices]

# Evolutionary Strategies ES algoritme
def evolutionary_strategies(returns, population_size, num_generations, mutation_rate):
    num_assets = returns.shape[1]
    population = initialize_population(population_size, num_assets)
    best_fitness_per_generation = []

    for generation in range(num_generations):
        # Steg 1: Evaluerer fitness
        fitness = np.array([fitness_function(ind, returns) for ind in population])

        # Steg 2: Selection (beholder beste halvdel av population)
        num_survivors = population_size // 2
        population = select_best(population, fitness, num_survivors)

        # Steg 3: rekombinasjon (generer offspring ved å rekombinere parents)
        offspring = []
        for _ in range(population_size - num_survivors):
            parents = np.random.choice(len(population), size=2, replace=False)
            child = recombine(population[parents])
            offspring.append(child)

        offspring = np.array(offspring)

        # Steg 4: Mutasjon 
        offspring = np.array([mutate(child, mutation_rate) for child in offspring])

        # Steg 5: lager ny population
        population = np.vstack((population, offspring))

        # Sporer beste fitness i hver generasjon 
        best_fitness_per_generation.append(np.max(fitness))

    best_portfolio = population[np.argmax(fitness)]
    return best_portfolio, best_fitness_per_generation

# Definerer parametere for Evolutionary Strategies algoritmer
population_size = 100  
num_generations = 500 
mutation_rate = 0.01  


# Kjører Evolutionary Strategies ES
best_portfolio_es, fitness_history_es = evolutionary_strategies(
    monthly_returns_df, population_size, num_generations, mutation_rate
)


# Evaluerer beste ES portfolio
best_portfolio_return_es = portfolio_return(best_portfolio_es, monthly_returns_df)
best_portfolio_risk_es = portfolio_risk(best_portfolio_es, covariance_matrix)
best_portfolio_volatility_es = np.sqrt(best_portfolio_risk_es)

# Printer ut ES portfolio resultater
print("\nBest Portfolio Expected Return (ES):", best_portfolio_return_es)
print("Best Portfolio Risk (Variance) (ES):", best_portfolio_risk_es)
print("Best Portfolio Volatility (Standard Deviation) (ES):", best_portfolio_volatility_es)

# Plot fitness history for ES
plt.plot(fitness_history_es)
plt.xlabel('Generation')
plt.ylabel('Best Fitness (Expected Return)')
plt.title('Evolution of Portfolio Expected Return (ES)')
plt.show()


