import random
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from IPython.display import Image, display

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

#random.seed(1)

def print_state(carcassonne_game_state: CarcassonneGameState):
    print_object = {
        "scores": {
            "player 1": carcassonne_game_state.scores[0],
            "player 2": carcassonne_game_state.scores[1]
        },
        "meeples": {
            "player 1": {
                "normal": str(carcassonne_game_state.meeples[0]) + " / " + str(carcassonne_game_state.meeples[0] + len(list(filter(lambda x: x.meeple_type == MeepleType.NORMAL or x.meeple_type == MeepleType.FARMER, game.state.placed_meeples[0])))),
                "abbots": str(carcassonne_game_state.abbots[0]) + " / " + str(carcassonne_game_state.abbots[0] + len(list(filter(lambda x: x.meeple_type == MeepleType.ABBOT, game.state.placed_meeples[0])))),
                "big": str(carcassonne_game_state.big_meeples[0]) + " / " + str(carcassonne_game_state.big_meeples[0] + len(list(filter(lambda x: x.meeple_type == MeepleType.BIG or x.meeple_type == MeepleType.BIG_FARMER, game.state.placed_meeples[0]))))
            },
            "player 2": {
                "normal": str(carcassonne_game_state.meeples[1]) + " / " + str(carcassonne_game_state.meeples[1] + len(list(filter(lambda x: x.meeple_type == MeepleType.NORMAL or x.meeple_type == MeepleType.FARMER, game.state.placed_meeples[1])))),
                "abbots": str(carcassonne_game_state.abbots[1]) + " / " + str(carcassonne_game_state.abbots[1] + len(list(filter(lambda x: x.meeple_type == MeepleType.ABBOT, game.state.placed_meeples[1])))),
                "big": str(carcassonne_game_state.big_meeples[1]) + " / " + str(carcassonne_game_state.big_meeples[1] + len(list(filter(lambda x: x.meeple_type == MeepleType.BIG or x.meeple_type == MeepleType.BIG_FARMER, game.state.placed_meeples[1]))))
            }
        }
    }

    print(print_object)

board_size = 20
game = CarcassonneGame(
    players=2,
    tile_sets=[TileSet.BASE],
    supplementary_rules=[],
    board_size=(board_size,board_size)
)

# turn on matplotlb interactive plotting
fig, ax = plt.subplots()
writer = PillowWriter(fps=16)
with writer.saving(fig, "carcassonne_game.gif", dpi=150):

    turn = 0
    while not game.is_finished():
        turn += 1
        
        # get game state
        player: int = game.get_current_player()
        valid_actions: [Action] = game.get_possible_actions()
        action: Optional[Action] = random.choice(valid_actions)
        
        # based on player AI
        if player==0:
            action = agent_random(valid_actions)
        elif player==1:
            action = agent_closest(valid_actions,game)

        # enact action
        if action is not None:
            game.step(player, action)

        # translate game state to array
        board_array = build_board_array(game,action)
        
        # show
        #game.render()
        plot_carcassonne_board(board_array, ax=ax)
        writer.grab_frame()
    
    # end game
    print_state(carcassonne_game_state=game.state)
# Display the saved GIF
display(Image(filename="carcassonne_game.gif"))