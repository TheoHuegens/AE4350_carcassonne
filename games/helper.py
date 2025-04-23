import random
import numpy as np
from typing import Optional
#import matplotlib.pyplot as plt

from wingedsheep.carcassonne.carcassonne_game import CarcassonneGame
from wingedsheep.carcassonne.carcassonne_game_state import CarcassonneGameState, GamePhase
from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.objects.actions.pass_action import PassAction
from wingedsheep.carcassonne.objects.actions.tile_action import TileAction
from wingedsheep.carcassonne.objects.actions.meeple_action import MeepleAction
from wingedsheep.carcassonne.objects.meeple_type import MeepleType
from wingedsheep.carcassonne.tile_sets.supplementary_rules import SupplementaryRule
from wingedsheep.carcassonne.tile_sets.tile_sets import TileSet

def build_board_array(game,action):
    
    # how to make array
    connection_region_dict = {
        'empty':            [[0,0,0],[0,0,0],[0,0,0]],
        'center':           [[0,0,0],[0,1,0],[0,0,0]],
        'bottom':           [[0,0,0],[0,0,0],[0,1,0]],
        'right':            [[0,0,0],[0,0,0],[0,0,1]],
        'top':              [[0,1,0],[0,0,0],[0,0,0]],
        'left':             [[0,0,0],[1,0,0],[0,0,0]],
        'bottomcenter':     [[0,0,0],[0,1,0],[0,1,0]],
        'bottomright':      [[0,0,0],[0,0,1],[0,1,1]],
        'bottomtop':        [[0,1,0],[0,1,0],[0,1,0]],
        'bottomleft':       [[0,0,0],[1,0,0],[1,1,0]],
        'rightcenter':      [[0,0,0],[0,1,1],[0,0,0]],
        'righttop':         [[0,1,1],[0,0,1],[0,0,0]],
        'rightleft':        [[0,0,0],[1,1,1],[0,0,0]],
        'rightbottom':      [[0,0,0],[0,0,1],[0,1,1]],
        'topcenter':        [[0,1,0],[0,1,0],[0,0,0]],
        'topright':         [[0,1,1],[0,0,1],[0,0,0]],
        'topleft':          [[1,1,0],[1,0,0],[0,0,0]],
        'topbottom':        [[0,1,0],[0,1,0],[0,0,0]],
        'leftcenter':       [[0,0,0],[1,1,0],[0,0,0]],
        'leftright':        [[0,0,0],[1,1,1],[0,0,0]],
        'lefttop':          [[0,1,0],[0,1,0],[0,1,0]],
        'leftbottom':       [[0,0,0],[1,0,0],[1,1,0]]
    }
    
    board_size = np.array(game.state.board).shape[0]
    board_array = np.zeros((10,3*board_size,3*board_size))
    for x in range(board_size):
        for y in range(board_size):
            
            tile = game.state.board[x][y]
            if tile is not None:
                tile_array = build_tile_array(tile,connection_region_dict)
                board_array[0:4,3*x:3*x+3,3*y:3*y+3] = tile_array
                
    return board_array

def build_tile_array(tile,connection_region_dict):
    
    tile_array = np.zeros((1,3,3)) # 3x3 subgrid for each tile property

    # make road
    tile_layer = np.zeros((3,3))
    for i in range(len(tile.road)):
        connection = tile.road[i].a.value+tile.road[i].b.value # from a to b
        tile_layer += connection_region_dict[connection]
    tile_array = np.append(tile_array,[tile_layer],axis=0)    
    
    # make city
    tile_layer = np.zeros((3,3))
    for i in range(len(tile.city)):
        castle = tile.city[i]
        for j in range(len(castle)):
            connection = castle[j].value
            tile_layer += connection_region_dict[connection]
    tile_array = np.append(tile_array,[tile_layer],axis=0)

    # make abbey
    if tile.chapel:
        tile_layer = connection_region_dict['center']
    else:
        tile_layer = connection_region_dict['empty']
    tile_array = np.append(tile_array,[tile_layer],axis=0)    

    return tile_array

def add_meeples_to_tile_array(game,tile_array,connection_region_dict):
    
    game.state.placed_meeples[0][0].coordinate_with_side.side.value
    for p in range(len(game.state.placed_meeples)): # meeples of player p
        # TODO: ensure own player is in same layer regarless of p,m 
        for m in range(len(game.state.placed_meeples[p])):
            meeple_placed = game.state.placed_meeples[p][m] # m th meeple of player p
            
            # extract info
            meeple_row = meeple_placed.coordinate_with_side.coordinate.row
            meeple_col = meeple_placed.coordinate_with_side.coordinate.column
            meeple_side = meeple_placed.coordinate_with_side.side.value