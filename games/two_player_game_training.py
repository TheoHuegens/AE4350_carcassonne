# module that runs a game between two agents

# first class imports
import random
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from IPython.display import Image, display
import contextlib
import time
# custom library imports
from wingedsheep.carcassonne.carcassonne_game import CarcassonneGame
from wingedsheep.carcassonne.carcassonne_game_state import CarcassonneGameState, GamePhase
from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.objects.actions.pass_action import PassAction
from wingedsheep.carcassonne.objects.actions.tile_action import TileAction
from wingedsheep.carcassonne.objects.actions.meeple_action import MeepleAction
from wingedsheep.carcassonne.objects.meeple_type import MeepleType
from wingedsheep.carcassonne.tile_sets.supplementary_rules import SupplementaryRule
from wingedsheep.carcassonne.tile_sets.tile_sets import TileSet
# local imports
from helper import *
from helper_plotting import *
from agents.agent import Agent
from agents.agents_alogirthmic import *
from agents.agents_ai import *
from two_player_game import two_player_game
from matplotlib.animation import PillowWriter

def play_multiple_games(
    N_games=2,
    board_size=15,
    max_turn=500,
    do_plot=True,
    player0_agent='random',
    player1_agent='random'
):
    score_histories = []
    reward_histories = []
    reward_cumul_histories = []
    network_frames = []

    for game_idx in range(N_games):
        print(f"\n--- Playing Game {game_idx+1}/{N_games} ---")
        random.seed(0) # can look at improvements better but worse for overall training -- testing only --
        random.seed(game_idx) # ensures reproducibility

        player_labels, score_history, rewards_history, rewards_cumul_history, losses, target_scores, predicted_scores, agent0, agent1, state_vector = two_player_game(
            board_size=board_size,
            max_turn=max_turn,
            do_plot=False, # never plot each game if doing many runs
            p0=player0_agent,
            p1=player1_agent,
            do_save=True,
            do_train=True,
            game_idx=game_idx
        )

        # Save final scores
        final_score = score_history[-1]
        score_histories.append(final_score)

        # Save final rewards
        final_reward = rewards_history[-1]
        reward_histories.append(final_reward)
        
        # Save final cumulative rewards
        final_cumul_reward = rewards_cumul_history[-1]
        reward_cumul_histories.append(final_cumul_reward)
    
        if do_plot:
            #plot_game_summary(player_labels, score_history, rewards_history, rewards_cumul_history, losses, target_scores, predicted_scores)
            fig_nn, _ = plot_network(agent0.policy_net, input_data=state_vector, game_idx=game_idx)
            
            # Capture frame as RGBA image for GIF
            fig_nn.canvas.draw()
            img = np.array(fig_nn.canvas.buffer_rgba())
            network_frames.append(img)
            plt.close(fig_nn)

    # Save network evolution GIF
    if do_plot and network_frames:
        fig_gif, ax = plt.subplots()
        writer = PillowWriter(fps=16)
        with writer.saving(fig_gif, "network_evolution.gif", dpi=400):
            for frame in network_frames:
                ax.imshow(frame)
                ax.axis('off')
                writer.grab_frame()
                ax.clear()
        plt.close(fig_gif)
        
        display(Image(filename="network_evolution.gif"))

    return score_histories, reward_histories

# Run N games
if __name__ == '__main__':
    p0 = "RLAgent"
    p1 = "score_max_potential_own"

    # train over N games
    N_games = 100
    score_histories, reward_histories = play_multiple_games(
        N_games=N_games,
        board_size=15,
        max_turn=500,
        do_plot=True,
        player0_agent=p0,
        player1_agent=p1
    )

    plot_training_progress(score_histories, reward_histories, player_labels={
        0: {"name": p0, "color": "blue"},
        1: {"name": p1, "color": "orange"},
    })
