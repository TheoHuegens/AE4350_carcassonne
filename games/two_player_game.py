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

def two_player_game(
    board_size,
    max_turn,
    do_convert,
    do_plot,
    player0_agent,
    player1_agent,
    gamma=0.99,
    reward_weights=(0.1, 1.0, 0.5, 10.0),
    record_rewards=True
):
    """
    Simulate a Carcassonne game between two agents, and train RL agents.

    Returns:
        score_history: list of scores over time
        rewards_history: list of (reward0, reward1) over time
    """
    starttime = time.time()

    game = CarcassonneGame(
        players=2,
        tile_sets=[TileSet.BASE],
        supplementary_rules=[],
        board_size=(board_size, board_size)
    )

    score_history = []
    rewards_history = []
    board_history = []
    player_history = []

    player_labels = {
        0: {"name": player0_agent.name, "color": "orange"},
        1: {"name": player1_agent.name, "color": "blue"}
    }

    if do_plot:
        fig, ax = plt.subplots()
        writer = PillowWriter(fps=16)
        writer_context = writer.saving(fig, "carcassonne_game.gif", dpi=150)
    else:
        fig, ax, writer, writer_context = None, None, None, contextlib.nullcontext()

    with writer_context:
        turn = 0
        while not game.is_finished() and turn < max_turn:
            turn += 1

            player = game.get_current_player()
            valid_actions = game.get_possible_actions()

            # Select action
            if player == 0:
                action = player0_agent.select_action(valid_actions, game, player)
            else:
                action = player1_agent.select_action(valid_actions, game, player)

            if action is not None:
                game.step(player, action)

            # Save board state
            if do_convert:
                board_array = build_board_array(game, do_norm=True)
                board_tensor = torch.tensor(board_array, dtype=torch.float32)
                board_tensor = torch.nan_to_num(board_tensor, nan=0.0)
                board_history.append(board_tensor)
                player_history.append(player)

                state_vector = build_state_vector(game)
                interpret_board_dict = interpret_board_array(board_array, state_vector)

                if do_plot:
                    plot_carcassonne_board(board_array, state_vector, player_labels, interpret_board_dict, ax=ax)
                    writer.grab_frame()

            # Record scores
            score_history.append(game.state.scores)

            # --- Immediate reward computation ---
            rewards = compute_rewards(
                score_history,
                game_finished=False,
                weights=reward_weights
            )
            rewards_history.append(rewards)

            # --- Immediate training ---
            if isinstance(player0_agent, RLAgent):
                player0_agent.train_step(board_tensor, target_score=rewards[0])
            if isinstance(player1_agent, RLAgent):
                player1_agent.train_step(board_tensor, target_score=rewards[1])

    # --- After game finished: discounted training pass ---
    T = len(board_history)
    final_rewards = compute_rewards(score_history, game_finished=True, weights=reward_weights)

    for t in range(T):
        discount = gamma ** (T - t - 1)
        board_tensor = board_history[t]
        player = player_history[t]

        if player == 0 and isinstance(player0_agent, RLAgent):
            player0_agent.train_step(board_tensor, target_score=final_rewards[0] * discount)
        if player == 1 and isinstance(player1_agent, RLAgent):
            player1_agent.train_step(board_tensor, target_score=final_rewards[1] * discount)

    # Save model at the end
    if isinstance(player0_agent, RLAgent):
        player0_agent.save_model()
    if isinstance(player1_agent, RLAgent):
        player1_agent.save_model()

    if do_plot:
        display(Image(filename="carcassonne_game.gif"))
        plot_score_history(score_history, player_labels)

    elapsed = time.time() - starttime
    print(f'Game took {elapsed:.2f} seconds')

    if record_rewards:
        return score_history, rewards_history
    else:
        return score_history

if __name__ == '__main__':

    # settings
    do_plot = True
    do_convert = True
    do_norm = True
    board_size = 15
    max_turn = 500

    # agents
    # dictionary mapping names to Agent **classes**
    agent_classes = {
        "random": RandomAgent(),
        "center": AgentCenter(),
        "score_max_potential_own": AgentScoreMaxPotentialOwn(),
        'human': AgentUserInput(),
        "RLAgent": RLAgent(),
    }
    player0_agent = agent_classes["center"]
    player1_agent = agent_classes["RLAgent"]

    # initial vars
    starttime = time.time()
    random.seed(1)
    
    # run game
    score_history = two_player_game(board_size,max_turn,do_convert,do_plot,player0_agent,player1_agent)
