import matplotlib.pyplot as plt

# Importere resultater fra både advanced EP og ES algoritmer
from portfolio_optimization_advanced_ep import (
    best_portfolio_return as best_portfolio_return_ep,
    best_portfolio_risk as best_portfolio_risk_ep,
    best_portfolio_volatility as best_portfolio_volatility_ep,
    fitness_history as fitness_history_ep
)

from portfolio_optimization_advanced_es import (
    best_portfolio_return_es,
    best_portfolio_risk_es,
    best_portfolio_volatility_es,
    fitness_history_es
)

# Plot fitness historikk for begge algoritmer
plt.plot(fitness_history_ep, label='Advanced EP Fitness')
plt.plot(fitness_history_es, label='Advanced ES Fitness')
plt.xlabel('Generation')
plt.ylabel('Best Fitness (Expected Return)')
plt.title('Comparison of Advanced EP and Advanced ES Fitness Over Generations')
plt.legend()
plt.show()

# Printer ut portfolio ytelses sammenligning
print("Comparison of Best Portfolio Metrics:")

print("\nAdvanced EP Results:")
print(f"Best Portfolio Expected Return (EP): {best_portfolio_return_ep}")
print(f"Best Portfolio Risk (Variance) (EP): {best_portfolio_risk_ep}")
print(f"Best Portfolio Volatility (Standard Deviation) (EP): {best_portfolio_volatility_ep}")

print("\nAdvanced ES Results:")
print(f"Best Portfolio Expected Return (ES): {best_portfolio_return_es}")
print(f"Best Portfolio Risk (Variance) (ES): {best_portfolio_risk_es}")
print(f"Best Portfolio Volatility (Standard Deviation) (ES): {best_portfolio_volatility_es}")
