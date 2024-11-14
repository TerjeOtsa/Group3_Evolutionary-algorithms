import numpy as np
import matplotlib.pyplot as plt

from portfolio_optimization_ep import fitness_history as ep_fitness_history
from portfolio_optimization_es import fitness_history_es as es_fitness_history
from portfolio_optimization_advanced_ep import fitness_history as advanced_ep_fitness_history
from portfolio_optimization_advanced_es import fitness_history_es as advanced_es_fitness_history
from es_combined_selection import fitness_history_es as mu_plus_lambda_fitness_history
from es_offspring_selection import fitness_history_es as mu_lambda_fitness_history


def convergence_speed(fitness_history, threshold_ratio=0.9):
    best_fitness = max(fitness_history)
    threshold_value = threshold_ratio * best_fitness
    for i, fitness in enumerate(fitness_history):
        if fitness >= threshold_value:
            return i 
    return len(fitness_history)  


def stability(fitness_history, last_n=100):
    if len(fitness_history) < last_n:
        last_n_fitness = fitness_history  
    else:
        last_n_fitness = fitness_history[-last_n:]
    return np.std(last_n_fitness)


fitness_histories = {
    "EP": ep_fitness_history,
    "ES": es_fitness_history,
    "Advanced EP": advanced_ep_fitness_history,
    "Advanced ES": advanced_es_fitness_history,
    "(μ + λ) ES": mu_plus_lambda_fitness_history,
    "(μ, λ) ES": mu_lambda_fitness_history
}


results = {}
for algorithm, history in fitness_histories.items():
    speed = convergence_speed(history)
    stability_value = stability(history)
    results[algorithm] = {
        "Convergence Speed (generations)": speed,
        "Stability (std dev)": stability_value
    }


print("\nComparison of Convergence Speed and Stability Across All Algorithms:\n")
for algorithm, metrics in results.items():
    print(f"{algorithm}:")
    print(f"  Convergence Speed: {metrics['Convergence Speed (generations)']} generations")
    print(f"  Stability: {metrics['Stability (std dev)']}")
    print()


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


algorithms = list(results.keys())
convergence_speeds = [results[algo]["Convergence Speed (generations)"] for algo in algorithms]
stabilities = [results[algo]["Stability (std dev)"] for algo in algorithms]


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(algorithms, convergence_speeds, color='skyblue')
plt.title("Convergence Speed of Each Algorithm")
plt.xlabel("Algorithm")
plt.ylabel("Generations to 90% of Best Fitness")
plt.xticks(rotation=45)


plt.subplot(1, 2, 2)
plt.bar(algorithms, stabilities, color='salmon')
plt.title("Stability of Each Algorithm")
plt.xlabel("Algorithm")
plt.ylabel("Standard Deviation of Last 100 Generations")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
