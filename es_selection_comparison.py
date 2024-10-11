import matplotlib.pyplot as plt

# Import the results from (μ + λ) and (μ, λ) ES algorithms
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

# Plot fitness histories for both algorithms
plt.plot(mu_plus_lambda_fitness_history, label='(μ + λ) ES Fitness')
plt.plot(mu_lambda_fitness_history, label='(μ, λ) ES Fitness')
plt.xlabel('Generation')
plt.ylabel('Best Fitness (Expected Return)')
plt.title('Comparison of (μ + λ) and (μ, λ) ES Fitness Over Generations')
plt.legend()
plt.show()

# Print the portfolio performance comparison
print("Comparison of Best Portfolio Metrics:")

print("\n(μ + λ) ES Results:")
print(f"Best Portfolio Expected Return (μ + λ): {mu_plus_lambda_return}")
print(f"Best Portfolio Risk (Variance) (μ + λ): {mu_plus_lambda_risk}")
print(f"Best Portfolio Volatility (Standard Deviation) (μ + λ): {mu_plus_lambda_volatility}")

print("\n(μ, λ) ES Results:")
print(f"Best Portfolio Expected Return (μ, λ): {mu_lambda_return}")
print(f"Best Portfolio Risk (Variance) (μ, λ): {mu_lambda_risk}")
print(f"Best Portfolio Volatility (Standard Deviation) (μ, λ): {mu_lambda_volatility}")
