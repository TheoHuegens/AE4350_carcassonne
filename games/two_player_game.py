# module that runs a game between two agents

import random
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from IPython.display import Image, display
import contextlib
import time

from wingedsheep.carcassonne.carcassonne_game import CarcassonneGame
from wingedsheep.carcassonne.carcassonne_game_state import CarcassonneGameState, GamePhase
from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.objects.actions.pass_action import PassAction
from wingedsheep.carcassonne.objects.actions.tile_action import TileAction
from wingedsheep.carcassonne.objects.actions.meeple_action import MeepleAction
from wingedsheep.carcassonne.objects.meeple_type import MeepleType
from wingedsheep.carcassonne.tile_sets.supplementary_rules import SupplementaryRule
from wingedsheep.carcassonne.tile_sets.tile_sets import TileSet

from actors import *
from helper import *

def two_player_game(
    board_size,max_turn,do_convert,do_plot, p0='agent_random',p1='agent_random'):

    # module that runs a game between two agents
    starttime = time.time()

    game = CarcassonneGame(
        players=2,
        tile_sets=[TileSet.BASE],
        supplementary_rules=[],
        board_size=(board_size, board_size)
    )

    score_history = []
    player_labels = {
        0: {"name": p0, "color": "orange"},
        1: {"name": p1, "color": "blue"}
    }

    # Only prepare plotting if do_plot is True
    if do_plot:
        fig, ax = plt.subplots()
        writer = PillowWriter(fps=16)
        writer_context = writer.saving(fig, "carcassonne_game.gif", dpi=150)
    else:
        fig, ax, writer, writer_context = None, None, None, contextlib.nullcontext()

    # Run game with optional plotting
    with writer_context:
        turn = 0
        while not game.is_finished() and turn<max_turn:
            turn += 1

            player = game.get_current_player()
            valid_actions = game.get_possible_actions()
            action = random.choice(valid_actions)

            # use an agent to choose action
            player_agent = player_labels[player]["name"]
            if player_agent=='agent_center':
                action = agent_center(valid_actions, game, player=player)
            elif player_agent=='agent_score_max_own':
                action = agent_score_max_own(valid_actions, game, player=player)
            elif player_agent=='agent_score_potential_max_own':
                action = agent_score_potential_max_own(valid_actions, game, player=player)
            elif player_agent=='agent_score_potential_max_gap':
                action = agent_score_potential_max_gap(valid_actions, game, player=player)
            elif player_agent=='agent_score_potential_delta_own':
                action = agent_score_potential_delta_own(valid_actions, game, player=player)
            elif player_agent=='agent_score_potential_delta_gap':
                action = agent_score_potential_delta_gap(valid_actions, game, player=player)
            else:
                action = agent_random(valid_actions, game, player=player)
                
            # enact action
            if action is not None:
                game.step(player, action)

            # Collect game state
            if do_convert:
                board_array = build_board_array(game,do_norm=False)
                state_vector = build_state_vector(game)
                score_history.append([state_vector[3], state_vector[4]])
                interpret_board_dict = interpret_board_array(board_array,state_vector)
            
                # Plot only if enabled
                if do_plot:
                    plot_carcassonne_board(board_array, state_vector, player_labels, interpret_board_dict, ax=ax)
                    writer.grab_frame()

            # in any case make note of the scores
            score_history.append(game.state.scores)

    # Show animation only if enabled
    if do_plot:
        display(Image(filename="carcassonne_game.gif"))
        plot_score_history(score_history, player_labels)

    elapsed = time.time() - starttime
    print(f'game took {elapsed} [s]')

    return score_history

if __name__ == '__main__':

    # settings
    do_plot = False
    do_convert = False
    do_norm = True
    board_size = 20
    max_turn = 500

    # agents
    p0='agent_center'
    p1='agent_center'

    # initial vars
    starttime = time.time()
    #random.seed(1)
    
    # run game
    score_history = two_player_game(board_size,max_turn,do_convert,do_plot,p0,p1)
