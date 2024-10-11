import matplotlib.pyplot as plt

# Import or re-run the EP and ES algorithms (assumes you've run both previously)
from portfolio_optimization_ep import best_portfolio_return, best_portfolio_risk, best_portfolio_volatility, fitness_history
from portfolio_optimization_es import best_portfolio_return_es, best_portfolio_risk_es, best_portfolio_volatility_es, fitness_history_es

# Print comparison of results
print("\nComparison between EP and ES:")
print("Best Portfolio Expected Return (EP):", best_portfolio_return)
print("Best Portfolio Risk (Variance) (EP):", best_portfolio_risk)
print("Best Portfolio Volatility (EP):", best_portfolio_volatility)

print("Best Portfolio Expected Return (ES):", best_portfolio_return_es)
print("Best Portfolio Risk (Variance) (ES):", best_portfolio_risk_es)
print("Best Portfolio Volatility (ES):", best_portfolio_volatility_es)

# Plot fitness histories for both algorithms
plt.plot(fitness_history, label='EP Fitness')
plt.plot(fitness_history_es, label='ES Fitness')
plt.xlabel('Generation')
plt.ylabel('Best Fitness (Expected Return)')
plt.title('Comparison of EP and ES Fitness Over Generations')
plt.legend()
plt.show()
