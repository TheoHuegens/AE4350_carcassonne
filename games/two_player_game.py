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
score_history = []
player_labels = {
    0: {"name":"agent_random ","color":"red"},
    1: {"name":"agent_closest","color":"blue"}
}

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
        state_vector = build_state_vector(game,action)
        score_history.append([state_vector[3],state_vector[4]])
        
        # show
        #game.render()
        plot_carcassonne_board(board_array,state_vector,player_labels, ax=ax)
        writer.grab_frame()
    
    # end game
    print_state(game)
# Display the saved GIF
display(Image(filename="carcassonne_game.gif"))

print_score_history(score_history,player_labels)