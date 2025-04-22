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
