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
    player1_agent
):
    """
    Simulate a Carcassonne game between two agent objects.

    Args:
        board_size: size of board
        max_turn: maximum number of turns
        do_convert: whether to create board arrays / state vectors
        do_plot: whether to plot the board state
        player0_agent: agent object for player 0
        player1_agent: agent object for player 1

    Returns:
        score_history: list of scores over time
    """
    
    starttime = time.time()

    game = CarcassonneGame(
        players=2,
        tile_sets=[TileSet.BASE],
        supplementary_rules=[],
        board_size=(board_size, board_size)
    )

    score_history = []
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

            # Choose action using the respective agent
            if player == 0:
                action = player0_agent.select_action(valid_actions, game, player)
            else:
                action = player1_agent.select_action(valid_actions, game, player)

            if action is not None:
                game.step(player, action)

            # Collect game state
            if do_convert:
                board_array = build_board_array(game, do_norm=False)
                state_vector = build_state_vector(game)
                score_history.append([state_vector[3], state_vector[4]])
                interpret_board_dict = interpret_board_array(board_array, state_vector)

                if do_plot:
                    plot_carcassonne_board(board_array, state_vector, player_labels, interpret_board_dict, ax=ax)
                    writer.grab_frame()

            # Always record scores
            score_history.append(game.state.scores)

    if do_plot:
        display(Image(filename="carcassonne_game.gif"))
        plot_score_history(score_history, player_labels)

    elapsed = time.time() - starttime
    print(f'Game took {elapsed:.2f} seconds')

    return score_history


if __name__ == '__main__':

    # settings
    do_plot = True
    do_convert = True
    do_norm = True
    board_size = 20
    max_turn = 500

    # agents
    # dictionary mapping names to Agent **classes**
    agent_classes = {
        "random": RandomAgent(Agent),
        #"center": CenterAgent,
        "score_max_potential_own": AgentScoreMaxPotentialOwn(Agent),
        #'agent_score_potential_max_own': AgentScoreMaxOwn(Agent),
        #"score_potential_max_gap": ScorePotentialMaxGapAgent,
        #"score_potential_delta_own": ScorePotentialDeltaOwnAgent,
        #"score_potential_delta_gap": ScorePotentialDeltaGapAgent,
    }
    p0 = agent_classes["random"]
    p1 = agent_classes["score_max_potential_own"]

    # initial vars
    starttime = time.time()
    random.seed(1)
    
    # run game
    score_history = two_player_game(board_size,max_turn,do_convert,do_plot,p0,p1)
