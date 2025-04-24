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

starttime = time.time()

do_plot = True  # Toggle this to False to skip plotting and speed things up
do_convert = True
do_norm = True
random.seed(1)
board_size = 20
max_turn = 500

game = CarcassonneGame(
    players=2,
    tile_sets=[TileSet.BASE],
    supplementary_rules=[],
    board_size=(board_size, board_size)
)

score_history = []
player_labels = {
    0: {"name": "agent_potential", "color": "orange"},
    1: {"name": "agent_closest  ", "color": "blue"}
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

        # AI decisions
        if player == 0:
            action = agent_score_potential_max_own(valid_actions, game, player=0)
        elif player == 1:
            action = agent_score_potential_delta_gap(valid_actions, game, player=1)

        if action is not None:
            game.step(player, action)

        # Collect game state
        if do_convert:
            board_array = build_board_array(game,do_norm=do_norm)
            state_vector = build_state_vector(game)
            score_history.append([state_vector[3], state_vector[4]])
            interpret_board_dict = interpret_board_array(board_array,state_vector)
        
        # Plot only if enabled
        if do_plot:
            plot_carcassonne_board(board_array, state_vector, player_labels, interpret_board_dict, ax=ax)
            writer.grab_frame()

    print_state(game)

# Show animation only if enabled
if do_plot:
    display(Image(filename="carcassonne_game.gif"))
    plot_score_history(score_history, player_labels)

elapsed = time.time() - starttime
print(f'game took {elapsed} [s]')
