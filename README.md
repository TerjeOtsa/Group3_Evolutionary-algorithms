### Required Libraries

To run this project, you’ll need the following Python libraries:

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations and handling arrays.
- **seaborn**: For data visualization and plotting.
- **deap**: A library for implementing evolutionary algorithms.
- **re**: Regular expressions, used for string manipulation.
- **scikit-learn**: For data preprocessing, including `MinMaxScaler` and `StandardScaler`.
- **pickle**: For object serialization and deserialization.
- **gym**: Part of OpenAI Gym, a toolkit for developing and comparing reinforcement learning algorithms. Provides various standard environments (such as `Taxi-v3` and `CartPole-v1`) for testing and training reinforcement learning models. 

### Installation

You can install these libraries using `pip`. Run the following command in your terminal:

```bash
pip install pandas numpy seaborn deap scikit-learn gym
```

The re and pickle libraries are built into Python, so no additional installation is needed for these.


## TASK 1:
# Traffic Management Optimization Using Multi-Objective Evolutionary Algorithms

This project optimizes traffic management using a Multi-Objective Evolutionary Algorithm (MOEA). The codebase is organized into five main files:

- **data_exploration.ipynb**: For data analysis and visualization.
- **dataset_manager.ipynb**: For preparing and processing datasets.
- **MOEA.ipynb**: For implementing the MOEA algorithm.
- **fuel_model.py**: For calculating fuel consumption based on traffic data.
- **segment_model.py**: For managing road segment data.

---

## Dataset Management

The **dataset_manager.ipynb** file formats raw data into smaller, more manageable subsets required for data exploration and the MOEA algorithm. However, running this file is **not necessary** since processed data is already stored as `.pickle` files in the `/datasets` folder.

### Datasets
- **Volume Data**: Located in `/datasets/VOL`.
- **Speed Data**: Not included in the repository due to its large size but can be fetched from:
  [NYC Traffic Speeds Dataset](https://data.cityofnewyork.us/Transportation/DOT-Traffic-Speeds-NBE/i4gi-tjb9/about_data).  
  The speed data has been queried for the last 3 years, but only the most recent year is used for simplicity.

### Running dataset_manager.ipynb
1. Place the speed dataset in the `/datasets/SPEED` folder.
2. Ensure the file paths and naming conventions match those used in the notebook.
3. Open the notebook and run all cells from the top using the built-in "Run All" feature or execute individual cells as needed.

---

## Running the Notebooks

### data_exploration.ipynb
- Purpose: Visualize and analyze the dataset.
- How to Run: Open the notebook and run all cells from the top or selectively execute cells.

### MOEA.ipynb
- Purpose: Run the Multi-Objective Evolutionary Algorithm.
- How to Run: Open the notebook and run all cells from the top or selectively execute cells.
- Configurations: If wanted the configurations parameters can be adjusted near the bottom of the file in a grid, under the "MOEA CONFIGURATIONS AND RESULTS" cell.

### Why Use Notebooks?
We chose notebooks for these tasks because they allow:
- Incremental execution of code cells, which is ideal for testing small changes.
- Time-efficient development by focusing on specific parts of the workflow.
- Immediate preview of images, tables, and print outputs beneath each cell.

---

## Supporting Python Files

- **fuel_model.py**: Contains functions for calculating fuel consumption per hour and per segment. These functions are imported as needed by the notebooks.
- **segment_model.py**: Manages segment-specific data, allowing for adjustments if required. It is also imported by the notebooks as needed. Spesific distances for the google maps calculations, they can be seen in datasets/QUERIES folder

> Note: These Python files do not need to be run independently but can be edited to customize calculations or segment details. A screenshot of the fuel calculations table is included for reference under the Fuel calculations table folder as a png.

---

With this structure, the project balances efficiency, modularity, and scalability, enabling easy exploration, adjustments, and optimization for traffic management solutions.






Comparative Analysis of Evolutionary Programming and Evolutionary Strategies for Portfolio Optimization:

This project contains Python scripts that implement different evolutionary algorithms, including Evolutionary Programming (EP) and Evolutionary Strategies (ES), to optimize a stock portfolio based on historical stock data. The results from both algorithms are compared to identify the best-performing portfolio in terms of expected return, risk, and volatility.

Files Overview:

portfolio_optimization_ep.py: Implements the Evolutionary Programming (EP) algorithm to optimize a portfolio of stocks.

portfolio_optimization_es.py: Implements the Evolutionary Strategies (ES) algorithm to optimize a portfolio of stocks.

portfolio_optimization_advanced_ep.py: Implements an advanced version of the EP algorithm with self-adaptive mutation strategies.

portfolio_optimization_advanced_es.py: Implements an advanced version of the ES algorithm with self-adaptive mutation rates and recombination.

es_combined_selection.py: Implements the (μ + λ) version of the ES algorithm, where the next generation is selected from both parent and offspring populations.

es_offspring_selection.py: Implements the (μ, λ) version of the ES algorithm, where the next generation is selected only from offspring, introducing greater selection pressure.

portfolio_comparison.py: Compares the performance of the portfolios optimized by both EP and ES in terms of expected return, risk, and volatility.

portfolio_comparison_advanced.py: Compares the advanced versions of EP and ES.

es_selection_comparison.py: Compares the (μ + λ) ES and (μ, λ) ES versions in terms of portfolio performance.

Setup Instructions:

Prerequisites
Before running any of the scripts, ensure you have Python installed, along with the following packages at the top (Required Libraries):

Data Requirements:
The scripts expect historical stock data and the covariance matrix to be present as CSV files in a folder. Ensure the following files are located in the specified directory:
Monthly Returns Data: Monthly_Returns_Data.csv
Covariance Matrix: Covariance_Matrix.csv

Make sure the file paths in each script match the location of your data files. You can update the data_folder variable in the scripts if necessary.
data_folder = '/path/to/your/data/'

How to Run the Scripts:

Running Evolutionary Programming (EP)
To run the Evolutionary Programming (EP) algorithm for portfolio optimization, execute the portfolio_optimization_ep.py script: python portfolio_optimization_ep.py
The script will:
Load the stock data.
Run the EP algorithm over multiple generations.
Print the best portfolio's expected return, risk (variance), and volatility (standard deviation).
Plot the fitness evolution (best portfolio return) over the generations.


Running Evolutionary Strategies (ES)
To run the Evolutionary Strategies (ES) algorithm for portfolio optimization, execute the portfolio_optimization_es.py 
script: python portfolio_optimization_es.py
This script works similarly to the EP script but uses the ES algorithm for optimization. It also prints and plots the results.

Running Advanced EP and ES
For more complex portfolio optimization, run the advanced versions:
Scripts:
portfolio_optimization_advanced_ep.py for Advanced EP.
portfolio_optimization_advanced_es.py for Advanced ES.
These versions include self-adaptive mutation strategies and advanced selection mechanisms.

Comparing the diffrent EP and ES versions
To compare the performance of the portfolios generated by the EP and ES algorithms, run the portfolio_full_comparison.py script
This script will:
Print the expected return, risk, and volatility of the portfolios optimized by the diffrent algorithms.
Plot the evolution of fitness (expected return) for all algorithms over the generations, allowing a visual comparison of how each algorithm performed.

 To compares the performance of the evolutionary algorithms in terms of convergence speed and stability based on their fitness histories
run the convergence_speed_stability.py script


Interpreting Results:
Expected Return: The higher, the better. It shows the average return of the portfolio over time.
Risk (Variance): The lower, the better. It indicates how much the portfolio's return can vary.
Volatility (Standard Deviation): The lower, the better. It represents the risk or uncertainty in the portfolio's return.
The comparison between EP and ES, and between different versions of ES, will help determine which algorithm performs better in balancing risk and return.

Customizing Parameters
You can adjust the following parameters in both the EP and ES scripts:

Population Size: The number of portfolios in each generation.
Number of Generations: The number of iterations the algorithm will run.
Mutation Rate: The probability and magnitude of mutations during the algorithm.
population_size = 100  # Number of portfolios in the population
num_generations = 500  # Number of generations to run the algorithm
mutation_rate = 0.01   # Probability of mutation




Task 4: Taxi-v3 Environment
Overview
Task_4.py: Trains the Q-learning agent.
run_trained_agent.py: Demonstrates the trained agent's performance.
Sarsa_compare.py: Compares the Q-learning (off-policy) agent with a SARSA (on-policy) agent, highlighting the differences in performance.

Dependencies
Ensure you have Python 3.x installed, along with the following packages:
gym (part of OpenAI Gym, installable via pip install gym)
numpy
matplotlib
seaborn

Training the Agent and Evaluating (Task_4.py)
This script trains a Q-learning agent on the Taxi-v3 environment.
Parameters
Learning Rate (alpha): Controls how much the agent updates the Q-values for each state-action pair. Set to 0.1.
Discount Factor (gamma): Determines the importance of future rewards. Set to 0.8.
Exploration Rate (epsilon): Balances exploration (random actions) and exploitation (greedy actions). Starts at 1.0 and decays over time.
Episodes (num_episodes): Maximum number of training episodes.
Early Stopping: Stops training if there’s no performance improvement over 50 intervals (each interval being 10 episodes).
Output
Q-table: The learned Q-table is saved to trained_q_table.npy for future testing.
Plots: The script generates multiple visualizations to illustrate the training process:
Total Rewards Over Time: Shows the accumulation of total rewards across episodes.
Average Reward Every 10 Episodes: Tracks average performance over each 10-episode interval.
Epsilon Decay Over Time: Visualizes the decay of the exploration rate throughout training.
Histogram of Total Rewards (Post-Training Evaluation): Displays the distribution of rewards per episode after training.

Testing the Trained Agent (Task_4_run_trained_model.py)
This script loads the trained Q-table from trained_q_table.npy and runs the agent in the Taxi-v3 environment to demonstrate its learned behavior.

Comparing Q-Learning and SARSA (Sarsa_compare.py)
The Sarsa_compare.py script runs a comparison between the Q-learning agent (off-policy) and a SARSA agent (on-policy). This allows for analysis of the differences in learning approaches and performance outcomes between these two reinforcement learning algorithms.
