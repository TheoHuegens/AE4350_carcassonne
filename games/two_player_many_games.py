# module that runs many games between many agents

# first class imports
import random
import numpy as np
import time
from multiprocessing import Pool, cpu_count
# local imports
from two_player_game import *

# === SETTINGS ===
do_plot = False
do_convert = False
do_norm = True
board_size = 20
max_turn = 50
games_per_match = 10
N_CORES = 1#cpu_count()  # Automatically use the number of CPU cores

# dictionary mapping names to Agent **classes**
agent_classes = {
    #"random": RandomAgent(),
    "center": AgentCenter,
    #"score_max_potential_own": AgentScoreMaxPotentialOwn(),
    #'human': AgentUserInput(),
    "RLAgent": RLAgent,
}

agents_to_test = list(agent_classes.keys())

# === SIMULATE MATCHUPS ===

def run_matchup(args):
    """
    Function that runs a matchup between two agents for a given number of games.
    This will be used for parallelization.
    """
    p0_name, p1_name, games_per_match = args
    win_counts_matrix = np.zeros((len(agents_to_test), len(agents_to_test)))
    score_gap_matrix = np.zeros((len(agents_to_test), len(agents_to_test)))

    # Instantiate agents
    p0_agent = agent_classes[p0_name]()
    p1_agent = agent_classes[p1_name]()

    for game_seed in range(games_per_match):
        random.seed(game_seed)
        score_history = two_player_game(board_size, max_turn, do_convert, False, p0_agent, p1_agent)

        final_score = score_history[-1]  # [p0_score, p1_score]
        p0_final, p1_final = final_score[0], final_score[1]
        score_gap_matrix[agents_to_test.index(p0_name), agents_to_test.index(p1_name)] += p0_final - p1_final

        if p0_final > p1_final:
            win_counts_matrix[agents_to_test.index(p0_name), agents_to_test.index(p1_name)] += 1
        elif p0_final == p1_final:
            win_counts_matrix[agents_to_test.index(p0_name), agents_to_test.index(p1_name)] += 0.5

    return win_counts_matrix, score_gap_matrix

# === MAIN EXECUTION ===
starttime = time.time()

# Prepare matchups
matchup_args = [(p0_name, p1_name, games_per_match) for i, p0_name in enumerate(agents_to_test) for j, p1_name in enumerate(agents_to_test)]

# Use multiprocessing to run all matchups
with Pool(N_CORES) as pool:
    results = pool.map(run_matchup, matchup_args)

# Aggregate results
win_counts_matrix = np.zeros((len(agents_to_test), len(agents_to_test)))
score_gap_matrix = np.zeros((len(agents_to_test), len(agents_to_test)))

for result in results:
    win_counts_matrix += result[0]
    score_gap_matrix += result[1]

# === COMPUTE AND PLOT WIN RATE MATRIX ===
win_rates = win_counts_matrix / games_per_match
score_gap_matrix /= games_per_match

plot_winrate_heatmap(win_rates, agents_to_test)
plot_scoregap_heatmap(score_gap_matrix, agents_to_test)

elapsed = time.time() - starttime
print(f"All matches completed in {elapsed:.2f} seconds.")
