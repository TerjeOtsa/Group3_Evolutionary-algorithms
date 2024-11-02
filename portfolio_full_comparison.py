import matplotlib.pyplot as plt

# Importerer resultater fra all algoritmer
from portfolio_optimization_ep import (
    best_portfolio_return as ep_return,
    best_portfolio_risk as ep_risk,
    best_portfolio_volatility as ep_volatility,
    fitness_history as ep_fitness_history
)

from portfolio_optimization_es import (
    best_portfolio_return_es as es_return,
    best_portfolio_risk_es as es_risk,
    best_portfolio_volatility_es as es_volatility,
    fitness_history_es as es_fitness_history
)

from portfolio_optimization_advanced_ep import (
    best_portfolio_return as advanced_ep_return,
    best_portfolio_risk as advanced_ep_risk,
    best_portfolio_volatility as advanced_ep_volatility,
    fitness_history as advanced_ep_fitness_history
)

from portfolio_optimization_advanced_es import (
    best_portfolio_return_es as advanced_es_return,
    best_portfolio_risk_es as advanced_es_risk,
    best_portfolio_volatility_es as advanced_es_volatility,
    fitness_history_es as advanced_es_fitness_history
)

from es_combined_selection import (
    best_portfolio_return_es as mu_plus_lambda_return,
    best_portfolio_risk_es as mu_plus_lambda_risk,
    best_portfolio_volatility_es as mu_plus_lambda_volatility,
    fitness_history_es as mu_plus_lambda_fitness_history
)

from es_offspring_selection import (
    best_portfolio_return_es as mu_lambda_return,
    best_portfolio_risk_es as mu_lambda_risk,
    best_portfolio_volatility_es as mu_lambda_volatility,
    fitness_history_es as mu_lambda_fitness_history
)

# Printer sammenligning av resultat av alle algoritmer
print("Comparison of Portfolio Performance Across All Algorithms:\n")

print("1. Evolutionary Programming (EP):")
print(f"Best Portfolio Expected Return: {ep_return}")
print(f"Best Portfolio Risk (Variance): {ep_risk}")
print(f"Best Portfolio Volatility (Standard Deviation): {ep_volatility}\n")

print("2. Evolutionary Strategies (ES):")
print(f"Best Portfolio Expected Return: {es_return}")
print(f"Best Portfolio Risk (Variance): {es_risk}")
print(f"Best Portfolio Volatility (Standard Deviation): {es_volatility}\n")

print("3. Advanced Evolutionary Programming (Advanced EP):")
print(f"Best Portfolio Expected Return: {advanced_ep_return}")
print(f"Best Portfolio Risk (Variance): {advanced_ep_risk}")
print(f"Best Portfolio Volatility (Standard Deviation): {advanced_ep_volatility}\n")

print("4. Advanced Evolutionary Strategies (Advanced ES):")
print(f"Best Portfolio Expected Return: {advanced_es_return}")
print(f"Best Portfolio Risk (Variance): {advanced_es_risk}")
print(f"Best Portfolio Volatility (Standard Deviation): {advanced_es_volatility}\n")

print("5. (μ + λ) ES:")
print(f"Best Portfolio Expected Return: {mu_plus_lambda_return}")
print(f"Best Portfolio Risk (Variance): {mu_plus_lambda_risk}")
print(f"Best Portfolio Volatility (Standard Deviation): {mu_plus_lambda_volatility}\n")

print("6. (μ, λ) ES:")
print(f"Best Portfolio Expected Return: {mu_lambda_return}")
print(f"Best Portfolio Risk (Variance): {mu_lambda_risk}")
print(f"Best Portfolio Volatility (Standard Deviation): {mu_lambda_volatility}\n")

# Plot fitness historikk for alle algoritmer
plt.plot(ep_fitness_history, label='EP Fitness')
plt.plot(es_fitness_history, label='ES Fitness')
plt.plot(advanced_ep_fitness_history, label='Advanced EP Fitness')
plt.plot(advanced_es_fitness_history, label='Advanced ES Fitness')
plt.plot(mu_plus_lambda_fitness_history, label='(μ + λ) ES Fitness')
plt.plot(mu_lambda_fitness_history, label='(μ, λ) ES Fitness')

plt.xlabel('Generation')
plt.ylabel('Best Fitness (Expected Return)')
plt.title('Comparison of Fitness Across All Algorithms')
plt.legend()
plt.show()
