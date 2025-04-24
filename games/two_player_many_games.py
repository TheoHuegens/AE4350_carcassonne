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

do_plot = True  # Toggle this to False to skip plotting and speed things up
board_size = 20
agents = ["agent_random","agent_closest","agent_highscore"]
player_labels = {
    0: {"name": agents[1], "color": "orange"},
    1: {"name": agents[1], "color": "blue"}
}

final_score_history = []

# Only prepare plotting if do_plot is True
if do_plot:
    fig, ax = plt.subplots()
    writer = PillowWriter(fps=16)
    writer_context = writer.saving(fig, "carcassonne_game.gif", dpi=150)
else:
    fig, ax, writer, writer_context = None, None, None, contextlib.nullcontext()

# Run games with optional plotting
for i in range(10):
    starttime = time.time()
    random.seed(i)
    turn = 0
    game = CarcassonneGame(
        players=2,
        tile_sets=[TileSet.BASE],
        supplementary_rules=[],
        board_size=(board_size, board_size)
    )

    score_history = []
    while not game.is_finished():
        turn += 1

        player = game.get_current_player()
        valid_actions = game.get_possible_actions()
        action = random.choice(valid_actions)

        # AI decisions
        if player == 0:
            if player_labels[player]["name"]=="agent_random":
                action = agent_random(valid_actions)
            if player_labels[player]["name"]=="agent_closest":
                action = agent_closest(valid_actions, game)
            if player_labels[player]["name"]=="agent_highscore":
                action = agent_highscore(valid_actions, game, player)
        elif player == 1:
            if player_labels[player]["name"]=="agent_random":
                action = agent_random(valid_actions)
            if player_labels[player]["name"]=="agent_closest":
                action = agent_closest(valid_actions, game)
            if player_labels[player]["name"]=="agent_highscore":
                action = agent_highscore(valid_actions, game, player)

        if action is not None:
            game.step(player, action)

        # Collect game state
        board_array = build_board_array(game)
        state_vector = build_state_vector(game)
        score_history.append([state_vector[3], state_vector[4]])

    #print_state(game)
    final_score_history.append(score_history)

    # Show last plot only if enabled
    if do_plot:
        plot_carcassonne_board(board_array, state_vector, player_labels, ax=None)
        plot_score_history(score_history, player_labels)

    elaped = time.time() - starttime
    print(f'game took {elaped} [s]')
    
plot_summary_results(final_score_history,player_labels)