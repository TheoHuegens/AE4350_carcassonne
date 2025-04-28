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
from agents.agent import Agent
from agents.agents_alogirthmic import *
from agents.agents_ai import *
from actors import *
from two_player_game import two_player_game

# --- New wrapper: Play many games ---
def play_multiple_games(
    N_games,
    board_size,
    max_turn,
    do_convert,
    do_plot_each,
    player0_agent,
    player1_agent
):
    score_histories = []
    reward_histories = []

    for game_idx in range(N_games):
        print(f"\n--- Playing Game {game_idx+1}/{N_games} ---")
        random.seed(game_idx) # ensures reproducibility
        #random.seed(0) # can look at improvements better but worse for overall training -- testing only --

        score_history, rewards_history = two_player_game(
            board_size=board_size,
            max_turn=max_turn,
            do_convert=do_convert,
            do_plot=do_plot_each,
            player0_agent=player0_agent,
            player1_agent=player1_agent,
            record_rewards=True
        )

        # Save final scores
        final_score = score_history[-1]
        score_histories.append(final_score)

        # Save final rewards
        final_reward = rewards_history[-1]
        reward_histories.append(final_reward)

    score_histories = np.array(score_histories)  # shape (N_games, 2)
    reward_histories = np.array(reward_histories)  # shape (N_games, 2)

    return score_histories, reward_histories

# Run N games
if __name__ == '__main__':
    # choose which of the possible agent that can play the game
    agent_classes = {
        "random": RandomAgent(),
        "center": AgentCenter(),
        "score_max_potential_own": AgentScoreMaxPotentialOwn(),
        'human': AgentUserInput(),
        "RLAgent": RLAgent(),
    }
    player0_agent=agent_classes["center"]
    player1_agent=agent_classes["RLAgent"]
    
    # setup training for both agents to use the same network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_policy_net = PolicyNet(input_channels=4).to(device)
    if isinstance(player0_agent, RLAgent):
        player0_agent = RLAgent(policy_net=shared_policy_net, model_path="shared_policy_net.pth")
    if isinstance(player1_agent, RLAgent):
        player1_agent = RLAgent(policy_net=shared_policy_net, model_path="shared_policy_net.pth")

    # train over N games
    N_games = 10
    score_histories, reward_histories = play_multiple_games(
        N_games,
        board_size=15,
        max_turn=50,
        do_convert=True,
        do_plot_each=False,
        player0_agent=player0_agent,
        player1_agent=player1_agent,
    )

    # and look at the results
    plot_training_progress(score_histories, reward_histories, player_labels={
        0: {"name": "Center", "color": "orange"},
        1: {"name": "RLAgent", "color": "blue"},
    })

