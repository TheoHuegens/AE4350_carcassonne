# first class imports
import random
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import label
from scipy.ndimage import convolve
from collections import deque
import seaborn as sns
import copy
import scipy.ndimage
# library imports
from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.objects.actions.pass_action import PassAction
from wingedsheep.carcassonne.objects.actions.tile_action import TileAction
from wingedsheep.carcassonne.objects.actions.meeple_action import MeepleAction
from wingedsheep.carcassonne.utils.points_collector import *

def construct_subtile_dict(do_norm=False):

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
    # Normalize the entries
    normalized_connection_region_dict = {}
    for key, mat in connection_region_dict.items():
        arr = np.array(mat, dtype=np.float32)
        total = arr.sum()
        if total > 0:
            arr /= total
        normalized_connection_region_dict[key] = arr  
    # use normed set or not  
    if do_norm:
        connection_region_dict = normalized_connection_region_dict

    return connection_region_dict

def build_board_array(game, do_norm=True, connection_region_dict=None):
    if connection_region_dict is None:
        connection_region_dict = construct_subtile_dict(do_norm=do_norm)

    board = game.state.board
    board_size = len(board)  # no need to np.array
    board_array = np.full((4, 3 * board_size, 3 * board_size), np.nan, dtype=np.float32)  # one-liner fill NaNs

    for x, row in enumerate(board):
        for y, tile in enumerate(row):
            if tile is not None:
                tile_array = build_tile_array(tile, connection_region_dict)
                sx, sy = 3 * x, 3 * y
                board_array[:, sx:sx+3, sy:sy+3] = tile_array

    placed_meeples = game.state.placed_meeples
    board_array = add_meeples_to_tile_array(placed_meeples, board_array, connection_region_dict)
    return board_array

def build_tile_array(tile,connection_region_dict):
    
    tile_array = np.zeros((4,3,3)) # 3x3 subgrid for each tile property

    # make road
    tile_layer = np.zeros((3,3))
    for i in range(len(tile.road)):
        connection = tile.road[i].a.value+tile.road[i].b.value # from a to b
        tile_layer += connection_region_dict[connection]
    tile_array[1] +=tile_layer
    
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
    tile_array[2] +=tile_layer

    # make abbey
    tile_layer = np.zeros((3,3))
    chapel_score = 1#connection_region_dict["full"][1][1] #0-1 since 8th neighbour tile = 9th tile = 1.125 but immediately earned before meeple turn
    if tile.chapel:
        # scale value with number of neighbours
        # TODO: find a way to this without the game object and whitout ommitting next tiles to be added to board array
        """
        neighbours = [ [0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1] ] # shift requires to get to 8 tiles around
        neighbour_num = chapel_score
        for n in neighbours:
            shifted_tile = game.state.board[x+n[0]][y+n[1]]
            if shifted_tile is not None:
                neighbour_num += chapel_score
        """
        neighbour_num=1
        tile_layer = neighbour_num * np.array(connection_region_dict['center'])
    else:
        tile_layer = connection_region_dict['empty']
    tile_array[3] += tile_layer

    return tile_array

def add_meeples_to_tile_array(placed_meeples,board_array,connection_region_dict):
    
    for p in range(len(placed_meeples)): # meeples of player p
        for m in range(len(placed_meeples[p])):
            meeple_placed = placed_meeples[p][m] # m th meeple of player p
            
            board_array = add_meeple_to_board_array(meeple_placed,p,board_array,connection_region_dict)
    
    return board_array # you'd think it's needed but no..

def add_meeple_to_board_array(meeple_placed,player,board_array,connection_region_dict):
    # TODO: ensure own player is in same layer regarless of p,m 
    if player == 0:  player_factor = +1 # own meeples
    else:       player_factor = -1 # opponent(s)'
    
    # extract info
    meeple_row = meeple_placed.coordinate_with_side.coordinate.row
    meeple_col = meeple_placed.coordinate_with_side.coordinate.column
    meeple_side = meeple_placed.coordinate_with_side.side.value
    
    # convert to board subgrid position and add on layer 0 (meeples)
    tile_layer = np.array(connection_region_dict[meeple_side]) * player_factor
    board_array[0,3*meeple_row:3*meeple_row+3,3*meeple_col:3*meeple_col+3] = tile_layer
    
    return board_array

def update_board_array(board_array,action,player,connection_region_dict=None,do_norm=False):
    
    if not isinstance(action,PassAction):
        # make dict if ont provided
        if connection_region_dict is None:
            connection_region_dict = construct_subtile_dict(do_norm=do_norm)

        # adds the tile on board
        if isinstance(action,TileAction):
            # get tile action info
            tile = action.tile 
            x=action.coordinate.column
            y=action.coordinate.row
            # translate to 3x3 subgrid board array coordinates
            sx, sy = 3 * x, 3 * y
            # make the tile
            tile_array = build_tile_array(tile,connection_region_dict)
            # add the tile to board
            board_array[:, sx:sx+3, sy:sy+3] = tile_array

        # adds the meeple on board
        if isinstance(action,MeepleAction):
            # only thing needed from meeple input is coordinate with side which is also in action
            board_array = add_meeple_to_board_array(action,player,board_array,connection_region_dict)
            
    return board_array

def build_state_vector(game_state,board_array_normed,board_array_bitwise):
    #board_array_bitwise = np.where(board_array_normed > 0, 1, np.where(board_array_normed < 0, -1, arr)) # faster to transform that way but idk if it works
    board_masks = interpret_board_array(board_array_bitwise) # ! un-normed !
    """
    "tile_mask": tile_mask,
    "road_mask": road_mask,
    "intersection_mask": intersection_mask,
    "connected_road_mask": connected_road_mask,
    "city_mask": city_mask,
    "city_shield_mask": city_shield_mask,
    "abbey_mask": abbey_mask,
    "meeples_mask_0": meeples_mask_mine,
    "meeples_mask_1": meeples_mask_oppo,
    "owned_mask_0": owned_mask_mine,
    "owned_mask_1": owned_mask_oppo
    """
    
    # game progress
    state_dict = {
        'tiles_left_in_drawpile': game_state[0]
    }
    
    # player attributes and status
    for p in range(2):
        #if p==0: factor=+1#0th player is default me
        #else:   factor =-1
        factor=1
        player_dict = {
            f'p{p}_score':               factor*float(game_state[1+2*p]),
            f'p{p}_meeple_in_hand':      factor*float(game_state[2+2*p]),
            f'p{p}_road_meeple':         factor*float(np.sum(board_masks['road_mask'] &   board_masks[f'meeples_mask_{p}'])), # meeples on a road
            f'p{p}_road_tiles':          factor*float(np.nansum(board_array_normed[1] *   board_masks[f'owned_mask_{p}'])), # all connected roads (excl. intersection) that overlap with the owned region, weighed
            f'p{p}_road_ends':           factor*float(0), # TBD
            f'p{p}_city_meeple':         factor*float(np.sum(board_masks['city_mask'] &   board_masks[f'meeples_mask_{p}'])),
            f'p{p}_city_tiles':          factor*float(np.nansum(board_array_normed[2] *   board_masks[f'owned_mask_{p}'])), # all connected cities that overlap with the owned region, weighed
            f'p{p}_city_ends':           factor*float(0), # TBD
            f'p{p}_abbeys':              factor*float(np.sum(board_masks['abbey_mask'] &  board_masks[f'meeples_mask_{p}'])),
            f'p{p}_abbey_neighbours':    factor*float(np.nansum(board_array_normed[3] *   board_masks[f'owned_mask_{p}'])) # weighed on 0-1 or 8 per abbey
        }
        state_dict.update(player_dict)
    
    state_vector = list(state_dict.values())
    state_vector = [x/100 for x in state_vector]
    #print(len(state_vector))
    
    return state_vector, state_dict

def find_contiguous_area(target_x, target_y, arr):
    """
    Return a boolean mask of the same shape as `arr`, where True indicates
    pixels that are part of the same 4-connected non-zero region as (target_y, target_x).
    
    Parameters:
        target_x (int): Column index (x)
        target_y (int): Row index (y)
        arr (ndarray): 2D array (e.g., mask of roads, cities, etc.)
    
    Returns:
        mask (ndarray): Boolean mask of the connected region
    """
    h, w = arr.shape
    if not (0 <= target_x < w and 0 <= target_y < h):
        raise ValueError("target_x and target_y must be within array bounds")

    if arr[target_y, target_x] == 0:
        return np.zeros_like(arr, dtype=bool)

    target_value = arr[target_y, target_x]
    visited = np.zeros_like(arr, dtype=bool)
    queue = deque([(target_y, target_x)])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connected (no diagonals)

    while queue:
        y, x = queue.popleft()
        if visited[y, x]:
            continue
        visited[y, x] = True

        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if not visited[ny, nx] and arr[ny, nx] != 0:
                    queue.append((ny, nx))

    return visited

def interpret_board_array(board_array):
    
    # read board array
    tile_mask =  ~np.isnan(board_array[1])
    road_mask = board_array[1] > 0
    intersection_mask = board_array[1] > 1
    connected_road_mask = ( road_mask ^ intersection_mask) == True
    city_mask = board_array[2] > 0
    city_shield_mask = board_array[2] > 1
    abbey_mask = board_array[3] > 0

    # read meeple positions
    meeples_mask_mine = board_array[0] > 0
    meeples_mask_oppo = board_array[0] < 0
    mine_y, mine_x = np.where(meeples_mask_mine)
    oppo_y, oppo_x = np.where(meeples_mask_oppo)

    # get contigous areas intersecting meeples per player
    # get all subtiles owned by my meeples
    owned_mask_mine = np.zeros_like(tile_mask).astype(bool)
    for m in range(len(mine_x)): # for every meeple placed
        x,y = mine_x[m],mine_y[m]
        owned_mask_mine |= find_contiguous_area(x, y, connected_road_mask).astype(bool)
        owned_mask_mine |= find_contiguous_area(x, y, city_mask).astype(bool)
        owned_mask_mine |= find_contiguous_area(x, y, abbey_mask).astype(bool)
    # get all subtiles owned by opponent's meeples
    owned_mask_oppo = np.zeros_like(tile_mask).astype(bool)
    for m in range(len(oppo_x)): # for every meeple placed
        x,y = oppo_x[m],oppo_y[m]
        owned_mask_oppo |= find_contiguous_area(x, y, connected_road_mask).astype(bool)
        owned_mask_oppo |= find_contiguous_area(x, y, city_mask).astype(bool)
        owned_mask_oppo |= find_contiguous_area(x, y, abbey_mask).astype(bool)

    interpret_board_dict = {
        "tile_mask": tile_mask,
        "road_mask": road_mask,
        "intersection_mask": intersection_mask,
        "connected_road_mask": connected_road_mask,
        "city_mask": city_mask,
        "city_shield_mask": city_shield_mask,
        "abbey_mask": abbey_mask,
        "meeples_mask_0": meeples_mask_mine,
        "meeples_mask_1": meeples_mask_oppo,
        "owned_mask_0": owned_mask_mine,
        "owned_mask_1": owned_mask_oppo
    }
    
    return interpret_board_dict

def estimate_potential_score(game,method='array'):
    
    # do it by meeple end-game grab
    # for some reason the copied game still impact all game objects
    if method=='object':
        game_endmove = copy.copy(game)
        PointsCollector.count_final_scores(game_endmove.state)
        scores=game_endmove.state.scores
    
    # do it via board interpreted form
    elif method=='array':
        subtile_dict = construct_subtile_dict(do_norm=True)
        game_state = game.state
        action_game_state = [len(game_state.deck),game_state.scores[0],game_state.meeples[0],game_state.scores[1],game_state.meeples[1]]
        board_array_normed = build_board_array(game, do_norm=True, connection_region_dict=subtile_dict)
        board_array_bitwise = build_board_array(game, do_norm=False, connection_region_dict=subtile_dict)
        action_vector_array, action_vector_dict = build_state_vector(action_game_state,board_array_normed,board_array_bitwise)

        # layers 1=roads, 2=city, 3=abbeys
        # own
        scores=[0,0]
        for p in range(2):
            scores[p] = action_vector_dict[f"p{p}_score"]+action_vector_dict[f"p{p}_road_tiles"]+action_vector_dict[f"p{p}_city_tiles"]+8*action_vector_dict[f"p{p}_abbeys"]
            scores[p]/100
    return scores

def compute_rewards(
    score_history,
    game_finished,
    weights
):
    # w0=score w1=increase w2=gap w3=win
    """
    Compute rewards for both players.

    Args:
        score_history: list of [score_p0, score_p1] after each turn
        game_finished: bool, True if game finished
        weights: tuple (w0, w1, w2, w3)

    Returns:
        rewards: list [reward_player0, reward_player1]
        winner: int 0 or 1 or None (tie)
    """

    w0, w1, w2, w3 = weights
    turn = len(score_history)
    max_turn = 144 # for current tile deck
    if game_finished:
        turn_correction_factor = w3#/(max_turn-turn+1)
    else:
        turn_correction_factor = 1

    if len(score_history) < 2:
        previous_scores = [0, 0]
    else:
        previous_scores = score_history[-2]
    current_scores = score_history[-1]

    rewards = [0.0, 0.0]

    # Determine winner (only at end)
    if game_finished:
        if current_scores[0] > current_scores[1]:
            winner = 0
        elif current_scores[1] > current_scores[0]:
            winner = 1
        else:
            winner = None
    else:
        winner = None

    for player in [0, 1]:
        opponent = 1 - player
        current_score = current_scores[player]
        score_diff = current_scores[player] - previous_scores[player]
        score_gap = current_scores[player] - current_scores[opponent]

        # Win/loss reward
        """
        if game_finished:
            if winner is None:
                game_result = 0.0
            elif winner == player:
                game_result = 1.0
            else:
                game_result = -1.0
        else:
            game_result = 0.0
        """
        game_result = 0.0 # dont use this anymore

        reward = (w0 * current_score) + (w1 * score_diff) + (w2 * score_gap) + (w3 * game_result)
        rewards[player] = reward*turn_correction_factor/100

    return rewards

def training_plan(game_idx):
    game_gap = 10
    policy_algo_init='score_max_own'
    reward_weights =(1.0, 0.0, 0.0, 0.0)
    # w0=score w1=increase w2=gap w3=final turn factor

    # step 0: focus on end game rewards
    param_dict={
        'policy_algo_init':policy_algo_init,
        'epsilon':0.01, # % random moves for exploration
        'gamma':0.0, # discount rate, high=future rewards matter
        'reward_weights':reward_weights
        # w0=score w1=increase w2=gap w3=final turn factor
    }
    # step 1: focus on end game rewards, increase future rewards
    """
    if 1*game_gap < game_idx:
        param_dict={
            'policy_algo_init':policy_algo_init,
            'epsilon':0.1, # % random moves for exploration
            'gamma':0.95, # discount rate, high=future rewards matter
            'reward_weights':reward_weights
            # w0=score w1=increase w2=gap w3=final turn factor
        }
    # step 2: stop exploration, increase future rewards
    if 2*game_gap < game_idx:
        param_dict={
            'policy_algo_init':policy_algo_init,
            'epsilon':0.01, # % random moves for exploration
            'gamma':0.975, # discount rate, high=future rewards matter
            'reward_weights':reward_weights
            # w0=score w1=increase w2=gap w3=final turn factor
        }
    """
    return param_dict