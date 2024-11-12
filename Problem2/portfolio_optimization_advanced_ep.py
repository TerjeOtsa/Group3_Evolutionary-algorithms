import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_folder = 'Problem2/'  

monthly_returns_df = pd.read_csv(data_folder + 'Monthly_Returns_Data.csv', index_col=0)
covariance_matrix = pd.read_csv(data_folder + 'Covariance_Matrix.csv', index_col=0)


num_stocks = len(monthly_returns_df.columns)  
population_size = 150  
num_generations = 500  
initial_mutation_rate = 0.1  
elitism_size = 3  

# Initialiserer population,tilfeldige weights

def initialize_population(size, num_assets):
    population = []
    mutation_rates = []  
    for _ in range(size):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)  
        population.append(weights)
        mutation_rates.append(initial_mutation_rate)  
    return np.array(population), np.array(mutation_rates)

# Fitness funksjon (forventet retur)
def fitness_function(weights, returns):
    return np.dot(weights, returns.mean())

# Mutasjon med self-adaptive strategi
def mutate(weights, mutation_rate):
    mutation = np.random.normal(0, mutation_rate, len(weights))  
    weights += mutation
    weights = np.clip(weights, 0, 1)  
    weights /= np.sum(weights) 
    return weights

# Definerer portfolio evaluerings funksjon
def portfolio_return(weights, returns):
    return np.dot(weights, returns.mean())

def portfolio_risk(weights, covariance_matrix):
    return np.dot(weights.T, np.dot(covariance_matrix, weights))

# Evolutionary Programming med Elitism og Self-Adaptive Mutasjon
def advanced_evolutionary_programming(returns, population_size, num_generations, elitism_size):
    population, mutation_rates = initialize_population(population_size, returns.shape[1])
    best_fitness_per_generation = []  

    for generation in range(num_generations):
        # regner ut fitness for nåværende population
        fitness = np.array([fitness_function(ind, returns) for ind in population])

        # Sorterer etter fitness for å implementere elitism
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        mutation_rates = mutation_rates[sorted_indices]
        fitness = fitness[sorted_indices]

        # beholder de beste individer (elitism)
        elites = population[-elitism_size:]

        # generer offspring gjennom mutasjon med self-adaptive mutasjons rater
        new_population = []
        new_mutation_rates = []
        for i in range(population_size - elitism_size):
            parent = population[i]
            mutation_rate = mutation_rates[i]

            # Self-adaptive mutasjon
            if i < population_size // 2:
                mutation_rate *= 0.95  # Senker mutasjons raten for øverste halvdel av population
            else:
                mutation_rate *= 1.2  # øker mutasjons raten for nederste halvdel av population

            offspring = mutate(parent, mutation_rate)
            new_population.append(offspring)
            new_mutation_rates.append(mutation_rate)

        # kombinerer elites og offspring til å lage en ny population
        population = np.vstack((new_population, elites))
        mutation_rates = np.array(new_mutation_rates + [initial_mutation_rate] * elitism_size)

        # sporer beste fitness i hver generasjon
        best_fitness_per_generation.append(np.max(fitness))

    # Returnerer beste portfolio og fitness historikk
    return population[np.argmax(fitness)], best_fitness_per_generation

# Kjører advanced EP algoritme
best_portfolio_weights, fitness_history = advanced_evolutionary_programming(monthly_returns_df, population_size, num_generations, elitism_size)

# Evaluerer beste portfolio funnet ved hjelp av algoritmen
best_portfolio_return = portfolio_return(best_portfolio_weights, monthly_returns_df)
best_portfolio_risk = portfolio_risk(best_portfolio_weights, covariance_matrix)
best_portfolio_volatility = np.sqrt(best_portfolio_risk)

# Printer portfolio ytelse
print("Best Portfolio Expected Return:", best_portfolio_return)
print("Best Portfolio Risk (Variance):", best_portfolio_risk)
print("Best Portfolio Volatility (Standard Deviation):", best_portfolio_volatility)

# Plot fitness evolusjon over generasjoner
plt.plot(fitness_history)
plt.xlabel('Generation')
plt.ylabel('Best Fitness (Expected Return)')
plt.title('Evolution of Portfolio Expected Return (Advanced EP)')
plt.show()

# Viser beste portfolio weights
print(best_portfolio_weights)