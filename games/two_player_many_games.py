import time
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from actors import *
from helper import *
from two_player_game import two_player_game

# === SETTINGS ===
do_plot = False
do_convert = False
do_norm = True
board_size = 15
max_turn = 50
games_per_match = 1

# === AGENTS ===
agents_to_test = ["agent_random", "agent_center", "agent_score_max_own","agent_score_potential_max_own","agent_score_potential_max_gap","agent_score_potential_delta_own","agent_score_potential_delta_gap"]

# === SIMULATE ALL MATCHUPS ===
starttime = time.time()
num_agents = len(agents_to_test)
win_counts_matrix = np.zeros((num_agents, num_agents))  # rows: player0, cols: player1
score_gap_matrix = np.zeros((num_agents, num_agents))

for i, p0 in enumerate(agents_to_test):
    for j, p1 in enumerate(agents_to_test):
        final_score_history = []

        for game_seed in range(games_per_match):
            random.seed(game_seed)
            score_history = two_player_game(board_size, max_turn, do_convert, False, p0, p1)

            if do_plot:
                player_labels = {
                    0: {"name": p0, "color": "orange"},
                    1: {"name": p1, "color": "blue"}
                }
                plot_score_history(score_history, player_labels)

            final_score = score_history[-1]
            final_score_history.append(final_score)
            score_gap_matrix[i, j] += final_score[0] - final_score[1]

            # Determine winner
            if final_score[0] > final_score[1]:
                win_counts_matrix[i, j] += 1
            elif final_score[0] == final_score[1]:
                win_counts_matrix[i, j] += 0.5  # count draws as half

        # Optional: print quick summary per matchup
        print(f"Match {p0} vs {p1}: {win_counts_matrix[i,j]}/{games_per_match} wins for {p0}")

# === COMPUTE AND PLOT WIN RATE MATRIX ===
win_rates = win_counts_matrix / games_per_match
score_gap_matrix /= games_per_match

plot_winrate_heatmap(win_rates, agents_to_test)
plot_scoregap_heatmap(score_gap_matrix, agents_to_test)

elapsed = time.time() - starttime
print(f"All matches completed in {elapsed:.2f} seconds.")
