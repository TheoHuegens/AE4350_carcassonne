import random
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import label
from scipy.ndimage import convolve
from collections import deque
import seaborn as sns
import copy
import scipy.ndimage
import torch
import torch.nn as nn
import os
import pandas as pd
import ast

from wingedsheep.carcassonne.utils.points_collector import *

def plot_scores_and_rewards(score_history, reward_history, player0_name="Player 0", player1_name="Player 1"):
    """
    Plot scores and rewards over time.

    Args:
        score_history: list of (score_p0, score_p1) at each turn
        reward_history: list of (reward_p0, reward_p1) at each turn
        player0_name: str
        player1_name: str
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # --- Scores ---
    axs[0].plot([s[0] for s in score_history], label=f"{player0_name} score", color='orange')
    axs[0].plot([s[1] for s in score_history], label=f"{player1_name} score", color='blue')
    axs[0].set_ylabel('Score')
    axs[0].set_xlabel('Turn')
    axs[0].set_title('Scores Over Time')
    axs[0].legend()
    axs[0].grid()

    # --- Rewards ---
    axs[1].plot([r[0] for r in reward_history], label=f"{player0_name} reward", color='orange')
    axs[1].plot([r[1] for r in reward_history], label=f"{player1_name} reward", color='blue')
    axs[1].set_ylabel('Reward')
    axs[1].set_xlabel('Turn')
    axs[1].set_title('Rewards Over Time')
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.show()

def moving_average(x, window_size=10):
    """Simple moving average for smoothing."""
    x = np.array(x)
    if len(x) < window_size:
        return x  # too short
    return np.convolve(x, np.ones(window_size) / window_size, mode='valid')

def plot_training_progress(score_histories, reward_histories, player_labels, smoothing_window=5):
    """
    Plot training curves: end scores and end rewards over games.
    
    Args:
        score_histories: list of [score0, score1] per game
        reward_histories: list of [reward0, reward1] per game
        player_labels: dict with {0: {name, color}, 1: {name, color}}
        smoothing_window: int, moving average window size
    """

    N_games = len(score_histories)
    
    # Extract scores and rewards directly
    final_scores_p0 = [scores[0] for scores in score_histories]
    final_scores_p1 = [scores[1] for scores in score_histories]

    final_rewards_p0 = [rewards[0] for rewards in reward_histories]
    final_rewards_p1 = [rewards[1] for rewards in reward_histories]

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    ## --- Plot final scores ---
    axs[0].plot(final_scores_p0, label=f"{player_labels[0]['name']} score", color=player_labels[0]['color'], alpha=0.4)
    axs[0].plot(final_scores_p1, label=f"{player_labels[1]['name']} score", color=player_labels[1]['color'], alpha=0.4)

    axs[0].plot(moving_average(final_scores_p0, smoothing_window), color=player_labels[0]['color'], lw=2)
    axs[0].plot(moving_average(final_scores_p1, smoothing_window), color=player_labels[1]['color'], lw=2)

    axs[0].set_title("Final Score per Game")
    axs[0].set_xlabel("Game")
    axs[0].set_ylabel("Score")
    axs[0].legend()
    axs[0].grid(True)

    ## --- Plot final rewards ---
    axs[1].plot(final_rewards_p0, label=f"{player_labels[0]['name']} reward", color=player_labels[0]['color'], alpha=0.4)
    axs[1].plot(final_rewards_p1, label=f"{player_labels[1]['name']} reward", color=player_labels[1]['color'], alpha=0.4)

    axs[1].plot(moving_average(final_rewards_p0, smoothing_window), color=player_labels[0]['color'], lw=2)
    axs[1].plot(moving_average(final_rewards_p1, smoothing_window), color=player_labels[1]['color'], lw=2)

    axs[1].set_title("Final Reward per Game")
    axs[1].set_xlabel("Game")
    axs[1].set_ylabel("Reward")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_game_summary(player_labels,score_history,rewards_history,rewards_cumul_history,losses,target_scores,predicted_scores):
    plot_score_history(score_history, player_labels,label='Score')
    plot_score_history(rewards_history, player_labels,label='Reward')
    plot_score_history(rewards_cumul_history, player_labels,label='Cumulative Reward')
    plot_score_history(target_scores, player_labels,label='Target Scores')
    plot_score_history(predicted_scores, player_labels,label='Predicted Scores')
    plot_score_history(losses, player_labels,label='Loss')

def plot_score_history(score_history,player_labels,label='Score'):
    plt.figure(figsize=(8, 4))
    plt.plot(np.array(score_history)[:,0], label=player_labels[0]['name'], color=player_labels[0]['color'], linewidth=2)
    plt.plot(np.array(score_history)[:,1], label=player_labels[1]['name'], color=player_labels[1]['color'], linewidth=2)
    plt.title(f"{label} Progression Over Game")
    plt.xlabel("Turn")
    plt.ylabel(label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # plot summary results
def plot_summary_results(final_score_history,player_labels):
    # Convert all score histories to a NumPy array for easier processing
    # Shape: (num_games, num_turns, num_players)
    # Since games may have different lengths, pad them
    max_turns = max(len(game_scores) for game_scores in final_score_history)
    num_players = len(final_score_history[0][0])

    # Initialize with NaNs for padding
    score_history_padded = np.full((len(final_score_history), max_turns, num_players), np.nan)

    for i, game_scores in enumerate(final_score_history):
        score_array = np.array(game_scores)
        score_history_padded[i, :len(score_array), :] = score_array

    # Normalize turns to percentage (0-100%)
    turns_percentage = np.linspace(0, 100, max_turns)

    # Calculate the mean score over time (across games)
    mean_score_over_time = np.nanmean(score_history_padded, axis=0)

    # Plot the mean scores over time (normalized to game %)
    plt.figure(figsize=(10, 5))
    for player_idx in range(num_players):
        plt.plot(turns_percentage, mean_score_over_time[:, player_idx], label=player_labels[player_idx]["name"],
                color=player_labels[player_idx]["color"])
    plt.xlabel("Game Progress (%)")
    plt.ylabel("Average Score")
    plt.title("Average Score Over Time (Normalized to Game %)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Extract final scores for boxplot (last score per game)
    final_scores = np.array([game_scores[-1] for game_scores in final_score_history])

    # Boxplot of final scores
    plt.figure(figsize=(6, 5))
    plt.boxplot(final_scores, labels=[player_labels[i]["name"] for i in range(num_players)])
    plt.ylabel("Final Score")
    plt.title("Final Scores Boxplot")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

def plot_final_scores_boxplot(final_score_history, player_labels=None, title="Final Score Distribution Over Games"):
    """
    Plots a boxplot of final scores for each player across multiple games.

    Args:
        final_score_history (list of list): Each inner list should contain scores of [player_1_score, player_2_score].
        player_labels (list of str, optional): Labels for the players. Default is ["Player 1", "Player 2"].
        title (str): Title of the plot.
    """
    # Default player labels
    if player_labels is None:
        player_labels = ["Player 1", "Player 2"]

    # Transpose the list to separate scores by player
    player_scores = list(zip(*final_score_history))

    # Plot setup
    plt.figure(figsize=(6, 4))
    box = plt.boxplot(
        player_scores,
        labels=player_labels,
        patch_artist=True,
        boxprops=dict(facecolor="skyblue", color="blue"),
        medianprops=dict(color="darkred"),
        whiskerprops=dict(color="blue"),
        capprops=dict(color="blue")
    )

    plt.title(title)
    plt.ylabel("Final Score")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_winrate_heatmap(win_matrix, agents):
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        100*win_matrix,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        cbar=True,
        xticklabels=agents,
        yticklabels=agents,
        annot_kws={"size": 12, "weight": "bold", "color": "black"},
        linewidths=0.5,
        linecolor='white'
    )

    ax.set_xlabel("Opponent", fontsize=14, weight="bold")
    ax.set_ylabel("Player", fontsize=14, weight="bold")
    ax.set_title("Win Rate Heatmap (percent)", fontsize=16, weight="bold")

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_scoregap_heatmap(score_gap_matrix, agents):
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        score_gap_matrix,
        annot=True,
        fmt=".1f",
        cmap="RdBu_r",  # diverging colormap: red = P0 loses, blue = P0 wins
        center=0,       # zero score gap in the middle
        xticklabels=agents,
        yticklabels=agents,
        annot_kws={"size": 12, "weight": "bold"},
        linewidths=0.5,
        linecolor='white'
    )

    ax.set_xlabel("Opponent", fontsize=14, weight="bold")
    ax.set_ylabel("Player", fontsize=14, weight="bold")
    ax.set_title("Average Score Gap (Rows vs Columns)", fontsize=16, weight="bold")

    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_carcassonne_board(board_array,state_dict,player_labels, interpret_board_dict, ax=None):
    # initialize plot
    if ax is None:
        fig, ax = plt.subplots()
    ax.clear()
    
    # Assuming board_array has shape (layers, height, width)
    height, width = board_array.shape[1], board_array.shape[2]

    # Start with background
    rgb_image = np.zeros((height, width, 3))
    background = 0.9
    rgb_image[:, :] = [background,background,background]
    # add each element
    rgb_image[interpret_board_dict["tile_mask"]] = [0,0.5,0]
    rgb_image[interpret_board_dict["road_mask"]] = [0.5,0.5,0.5]
    rgb_image[interpret_board_dict["intersection_mask"]] = [0.75,0.75,0.75]
    rgb_image[interpret_board_dict["city_mask"]] = [0,0,0.5]
    rgb_image[interpret_board_dict["city_shield_mask"]] = [0,0,0.75]
    rgb_image[interpret_board_dict["abbey_mask"]] = [1,0,0]
    ax.imshow(rgb_image)
    
    # Add hashed overlays using contourf hatching
    for idx, label in player_labels.items():
        key = f"owned_mask_{idx}"
        owned_mask = interpret_board_dict[key]
        color = label['color']
        if idx==0:
            hatches = ['', '----']
        else:
            hatches = ['', '||||']

        # Use contourf with hatching to overlay the owned region
        ax.contourf(
            owned_mask.astype(int),
            levels=1,
            hatches=hatches,
            alpha=0.,  # fully transparent fill
            colors=[color],  # hatch color
            zorder=5
        )
        # Set hatch color manually (defaults to black otherwise)
        #for collection in contours.collections:
        #    collection.set_edgecolor(color)
        #    collection.set_linewidth(0.5)  # optional: finer lines

                    
    # add meeples - Layer 0
    mine_y, mine_x = np.where(interpret_board_dict["meeples_mask_0"])
    oppo_y, oppo_x = np.where(interpret_board_dict["meeples_mask_1"])
    ax.plot(mine_x, mine_y, marker='x', linestyle='None', color=player_labels[0]['color'], markersize=5, markeredgewidth=2,zorder=6)
    ax.plot(oppo_x, oppo_y, marker='x', linestyle='None', color=player_labels[1]['color'], markersize=5, markeredgewidth=2,zorder=6)
    
    # add text for vector infos
    cards_left = state_dict["tiles_left_in_drawpile"]
    score_0 = state_dict["p0_score"]
    meeples_0 = state_dict["p0_meeple_in_hand"]
    score_1 =state_dict["p1_score"]
    meeples_1 = state_dict["p1_meeple_in_hand"]
    
    ax.set_title(f"Turns Left: {cards_left}", fontsize=12, weight='bold')

    # Bottom annotation for player info
    ax.text(0.01, -0.06,
            f"[P1] {player_labels[0]['name']}   Score: {score_0}    Meeples: {meeples_0}",
            transform=ax.transAxes, fontsize=9, color=player_labels[0]['color'], family='monospace')
    ax.text(0.01, -0.12,
            f"[P2] {player_labels[1]['name']}   Score: {score_1}    Meeples: {meeples_1}",
            transform=ax.transAxes, fontsize=9, color=player_labels[1]['color'], family='monospace')
    
    # Add dashed black grid lines every 3 cells
    offset = 0.6
    for x in range(0, width, 3):
        ax.axvline(x - offset, color='black', linestyle='--', linewidth=0.5)
    for y in range(0, height, 3):
        ax.axhline(y - offset, color='black', linestyle='--', linewidth=0.5)

    ax.axis("off")
    return ax

def plot_network(model, input_data=None, output_label="t",
                 layer_spacing=1.5, neuron_spacing=0.5, neuron_size=0.2,
                 weight_threshold=0.01, game_idx=0):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    act_color_map = {
        "Sigmoid": "#66ccff",
        "ReLU": "#66ff66",
        "Tanh": "#b266ff",
        "Softplus": "#ffcc66",
        "Linear": "#ffffff",
        "Input": "#add8e6"
    }

    # Structure extraction
    layers = []
    weights, biases, activations = [], [], []
    input_dim = None

    for m in model.children():
        if isinstance(m, nn.Linear):
            if input_dim is None:
                input_dim = m.in_features
                layers = [input_dim]
            layers.append(m.out_features)
            weights.append(m.weight.detach().cpu().numpy())
            biases.append(m.bias.detach().cpu().numpy())
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softplus, nn.Identity)):
            activations.append(m.__class__.__name__)

    input_labels = list(input_data.keys()) if input_data else [f"x{i}" for i in range(input_dim)]

    fig, ax = plt.subplots(figsize=(layer_spacing * len(layers), neuron_spacing * max(layers) * 1.1))
    neuron_positions = [None] * len(layers)

    # Draw neurons and record positions
    for layer_idx, layer_size in enumerate(layers):
        x = layer_idx * layer_spacing
        y_offset = (layer_size - 1) * neuron_spacing / 2
        color = (
            act_color_map["Input"] if layer_idx == 0 else
            act_color_map["Linear"] if layer_idx == len(layers) - 1 else
            act_color_map.get(activations[layer_idx - 1], "#dddddd")
        )

        layer_positions = []
        for neuron_idx in range(layer_size):
            y = -neuron_idx * neuron_spacing + y_offset
            layer_positions.append((x, y))
            circle = plt.Circle((x, y), neuron_size, color=color, ec='black', zorder=2)
            ax.add_patch(circle)

            if layer_idx == 0:
                label = input_labels[neuron_idx] if neuron_idx < len(input_labels) else f"x{neuron_idx}"
                ax.text(x - 0.4, y, label, va='center', ha='right', fontsize=9)
            elif layer_idx == len(layers) - 1:
                ax.text(x + 0.4, y, output_label, va='center', ha='left', fontsize=10)
            else:
                ax.text(x, y + 0.35, f"h{layer_idx}_{neuron_idx}", va='bottom', ha='center', fontsize=8)

            if layer_idx > 0:
                b_val = biases[layer_idx - 1][neuron_idx]
                ax.text(x, y - 0.35, f"b={b_val:.2f}", va='top', ha='center', fontsize=7, color='gray')

        neuron_positions[layer_idx] = layer_positions

    # Draw weighted connections
    for layer_idx, w in enumerate(weights):
        pos_in = neuron_positions[layer_idx]
        pos_out = neuron_positions[layer_idx + 1]

        rows, cols = np.where(np.abs(w) > weight_threshold)
        for j, i in zip(rows, cols):  # j = out, i = in
            x1, y1 = pos_in[i]
            x2, y2 = pos_out[j]
            weight = w[j, i]
            color = 'red' if weight < 0 else 'blue'
            lw = 0.5 + 2 * min(1, abs(weight))
            ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=0.6, zorder=1)

    # Add title and legend
    ax.set_title(f"Neural Network Architecture\n at training step {game_idx}", fontsize=14, pad=20)
    ax.axis('equal')
    ax.axis('off')

    legend_elements = [
        mpatches.Patch(color='red', label='Negative Weight'),
        mpatches.Patch(color='blue', label='Positive Weight')
    ] + [mpatches.Patch(color=col, label=f"{act} Neuron") for act, col in act_color_map.items()]

    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=8)

    plt.tight_layout()
    return fig, ax

def plot_stats(output_dir, labels, data_arr, title, ylabel, filename):
    """
    Plots the median line with a shaded confidence interval (25th–75th percentile).
    """
    #print(data_arr)
    data_arr = np.stack(data_arr)
    #print(data_arr.shape)
    #median = np.median(data_arr, axis=1)
    #q25 = np.percentile(data_arr, 25, axis=1)
    #q75 = np.percentile(data_arr, 75, axis=1)
    #episodes = range(data_arr.shape[0])
    
    plt.figure()
    for p in range(2):
        name = labels[p]['name']
        #print(data_arr[:,p])
        plt.plot(data_arr[:,p], label=f"{name} (Median)", color=labels[p]["color"])
        #plt.plot(median[:,p], label=f"{name} (Median)", color=labels[p]["color"])
        #plt.fill_between(episodes, q25[:,p], q75[:,p], color=labels[p]["color"], alpha=0.2, label='25th-75th percentile')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.legend()
    #plt.grid(True)
    os.makedirs(os.path.join(output_dir, 'fig'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'fig', filename))
    plt.close()
