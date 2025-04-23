import random
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import label

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
        # empty (different from no tile) and full tile 
        'empty':            [[0,0,0],[0,0,0],[0,0,0]],
        # meeple or city position
        'center':           [[0,0,0],[0,1,0],[0,0,0]],
        'bottom':           [[0,0,0],[0,0,0],[0,1,0]],
        'right':            [[0,0,0],[0,0,1],[0,0,0]],
        'top':              [[0,1,0],[0,0,0],[0,0,0]],
        'left':             [[0,0,0],[1,0,0],[0,0,0]],
        # farmer sub-position
        "top_left":         [[1,0,0],[0,0,0],[0,0,0]],
        "top_right":        [[0,0,1],[0,0,0],[0,0,0]],
        "bottom_left":      [[0,0,0],[0,0,0],[1,0,0]],
        "bottom_right":     [[0,0,0],[0,0,0],[0,0,1]],
        # network connection (2-sided)
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
        'topbottom':        [[0,1,0],[0,1,0],[0,1,0]],
        'leftcenter':       [[0,0,0],[1,1,0],[0,0,0]],
        'leftright':        [[0,0,0],[1,1,1],[0,0,0]],
        'lefttop':          [[1,1,0],[1,0,0],[0,0,0]],
        'leftbottom':       [[0,0,0],[1,0,0],[1,1,0]],
        # special connection cases of cities extent (3-4 sided)
        'not_bottom':       [[1,1,1],[1,1,1],[1,0,1]],
        'not_right':        [[1,1,1],[1,1,0],[1,1,1]],
        'not_top':          [[1,0,1],[1,1,1],[1,1,1]],
        'not_left':         [[1,1,1],[0,1,1],[1,1,1]],
        'full':             [[1,1,1],[1,1,1],[1,1,1]]
    }
    
    board_size = np.array(game.state.board).shape[0]
    board_array = np.zeros((10,3*board_size,3*board_size))
    board_array[:] = np.nan
    for x in range(board_size):
        for y in range(board_size):
            tile = game.state.board[x][y]
            if tile is not None:
                tile_array = build_tile_array(tile,game,x,y,connection_region_dict)
                board_array[0:4,3*x:3*x+3,3*y:3*y+3] = tile_array
                
    add_meeples_to_tile_array(game,board_array,connection_region_dict)
    
    return board_array

def build_tile_array(tile,game,x,y,connection_region_dict):
    
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
        # add each casle part (up to 4 separate domains per tile)
        connection = ''
        for j in range(len(castle)):
            connection += castle[j].value
        if len(castle) < 3: # castle has one or two sides
            tile_layer += connection_region_dict[connection]
        elif len(castle) == 4: # castle is full
            tile_layer += connection_region_dict['full']
        else:# castle is all except one side
            # get opposite of leftbottomtop = right
            sides = list(tile.get_city_sides())
            sides = sides[0].value+sides[1].value+sides[2].value
            if 'bottom' not in sides: connection = 'not_bottom'
            if 'right' not in sides: connection = 'not_right'
            if 'top' not in sides: connection = 'not_top'
            if 'left' not in sides: connection = 'not_left'
            tile_layer += connection_region_dict[connection]
        # double values if shield is present
        if tile.shield:
            tile_layer += tile_layer
    tile_array = np.append(tile_array,[tile_layer],axis=0)

    # make abbey
    tile_layer = np.zeros((3,3))
    if tile.chapel:
        # scale value with number of neighbours
        neighbours = [ [0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1] ] # shift requires to get to 8 tiles around
        neighbour_num = 1
        for n in neighbours:
            shifted_tile = game.state.board[x+n[0]][y+n[1]]
            if shifted_tile is not None:
                neighbour_num+=1
        tile_layer = neighbour_num * np.array(connection_region_dict['center'])

    else:
        tile_layer = connection_region_dict['empty']
    tile_array = np.append(tile_array,[tile_layer],axis=0)

    return tile_array

def add_meeples_to_tile_array(game,board_array,connection_region_dict):
    
    for p in range(len(game.state.placed_meeples)): # meeples of player p
        for m in range(len(game.state.placed_meeples[p])):
            meeple_placed = game.state.placed_meeples[p][m] # m th meeple of player p
            
            # TODO: ensure own player is in same layer regarless of p,m 
            if p ==0:   player_factor = +1 # own meeples
            else:       player_factor = -1 # opponent(s)'
            
            # extract info
            meeple_row = meeple_placed.coordinate_with_side.coordinate.row
            meeple_col = meeple_placed.coordinate_with_side.coordinate.column
            meeple_side = meeple_placed.coordinate_with_side.side.value
            
            # convert to board subgrid position and add on layer 0 (meeples)
            tile_layer = np.array(connection_region_dict[meeple_side]) * player_factor
            board_array[0,3*meeple_row:3*meeple_row+3,3*meeple_col:3*meeple_col+3] = tile_layer


def find_contigous_area(target_x, target_y, arr):

    # Define an 8-connected structure for labeling
    structure = np.ones((3, 3), dtype=int)

    # Label connected regions
    labeled_array, num_features = label(arr > 0, structure=structure)

    # Get label value at the chosen pixel
    target_label = labeled_array[target_y, target_x]

    # Create a mask for all pixels connected to the target
    connected_mask = labeled_array == target_label
    connected_mask = connected_mask.astype(int)
    
    # Optional: visualize or use the mask
    #print("Connected region to ({}, {}):".format(target_y, target_x))
    #print(connected_mask)
    
    return connected_mask

def plot_carcassonne_board(board_array, ax=None):
    # initialize plot
    if ax is None:
        fig, ax = plt.subplots()
    ax.clear()

    # Assuming board_array has shape (layers, height, width)
    height, width = board_array.shape[1], board_array.shape[2]

    # Start with no background
    rgb_image = np.zeros((height, width, 3))
    background = 0.9
    rgb_image[:, :] = [background,background,background]

    # Field (any tile) color (green)
    tile_mask =  ~np.isnan(board_array[1])
    rgb_image[tile_mask] = [0,0.5,0]

    # Roads (grey) - Layer 1
    road_mask = board_array[1] > 0
    intersection_mask = board_array[1] > 1
    connected_road_mask = ( road_mask ^ intersection_mask) == True
    rgb_image[road_mask] = [0.5,0.5,0.5]
    rgb_image[intersection_mask] = [0.75,0.75,0.75]

    # Cities (blue) - Layer 2
    city_mask = board_array[2] > 0
    rgb_image[city_mask] = [0,0,0.5]
    city_shield_mask = board_array[2] > 1
    rgb_image[city_shield_mask] = [0,0,0.75]

    # Abbeys (red) - Layer 3
    abbey_mask = board_array[3] > 0
    for s in range(8):
        abbey_score_mask = board_array[3] > s
        rgb_image[abbey_score_mask] = [s/8,0,0]
        
    # add meeples - Layer 0
    meeples_mask_mine = board_array[0] > 0
    meeples_mask_oppo = board_array[0] < 0
    mine_y, mine_x = np.where(meeples_mask_mine)
    oppo_y, oppo_x = np.where(meeples_mask_oppo)

    # show board
    ax.imshow(rgb_image)
    
    # show meeples
    ax.plot(mine_x, mine_y, marker='x', linestyle='None', color='yellow', markersize=5, markeredgewidth=2)
    ax.plot(oppo_x, oppo_y, marker='x', linestyle='None', color='magenta', markersize=5, markeredgewidth=2)
    
    # Add dashed black grid lines every 3 cells
    offset = 0.6
    for x in range(0, width, 3):
        ax.axvline(x - offset, color='black', linestyle='--', linewidth=0.5)
    for y in range(0, height, 3):
        ax.axhline(y - offset, color='black', linestyle='--', linewidth=0.5)

    ax.axis("off")