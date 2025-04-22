import random
import numpy as np
from typing import Optional

from wingedsheep.carcassonne.carcassonne_game import CarcassonneGame
from wingedsheep.carcassonne.carcassonne_game_state import CarcassonneGameState, GamePhase
from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.objects.actions.pass_action import PassAction
from wingedsheep.carcassonne.objects.actions.tile_action import TileAction
from wingedsheep.carcassonne.objects.actions.meeple_action import MeepleAction
from wingedsheep.carcassonne.objects.meeple_type import MeepleType
from wingedsheep.carcassonne.tile_sets.supplementary_rules import SupplementaryRule
from wingedsheep.carcassonne.tile_sets.tile_sets import TileSet

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

def agent_random(valid_actions):
    # this plays at random
    
    action: Optional[Action] = random.choice(valid_actions)
    return action

def agent_closest(valid_actions):
    # this plays closest to the center
    target = [35/2,35/2]
    
    action_scores = []
    for action_index in range(len(valid_actions)):
        score = 0
        action = valid_actions[action_index]
        # what's the action
        if isinstance(action,PassAction):
            #print('pass')
            dist = 30**2+30**2 # max
        elif isinstance(action,TileAction):
            #print('TILE: tile description: ', action.tile.description, ' - position: ',[action.coordinate.column,action.coordinate.row])
            dist = (action.coordinate.column-target[0])**2 + (action.coordinate.row-target[1])**2
        elif isinstance(action,MeepleAction):
            #print('MEEPLE: meeple type: ',action.meeple_type.value, ' - position: ', [action.coordinate_with_side.coordinate.column,action.coordinate_with_side.coordinate.row], ' - side: ', action.coordinate_with_side.side.value)
            dist = action.coordinate_with_side.coordinate.column**2 + action.coordinate_with_side.coordinate.row**2
        # score it
        action_scores.append(dist)
    #print(action_scores)
    action_index = np.argmin(action_scores)
    #print(action_index,len(valid_actions),action_scores[action_index])
    action = valid_actions[action_index]
    
    return action

game = CarcassonneGame(
    players=2,
    tile_sets=[TileSet.BASE],
    supplementary_rules=[]
)

while not game.is_finished():
    player: int = game.get_current_player()
    valid_actions: [Action] = game.get_possible_actions()
    action: Optional[Action] = random.choice(valid_actions)
    
    # based on player AI
    if player==0:
        action = agent_closest(valid_actions)
    elif player==1:
        action = agent_closest(valid_actions)

    if action is not None:
        game.step(player, action)
    game.render()

print_state(carcassonne_game_state=game.state)
