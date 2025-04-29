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
                tile_array = build_tile_array(tile, game, x, y, connection_region_dict)
                sx, sy = 3 * x, 3 * y
                board_array[:, sx:sx+3, sy:sy+3] = tile_array

    add_meeples_to_tile_array(game, board_array, connection_region_dict)
    return board_array

def build_tile_array(tile,game,x,y,connection_region_dict):
    
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
    tile_array[3] += tile_layer

    return tile_array

def add_meeples_to_tile_array(game,board_array,connection_region_dict):
    
    for p in range(len(game.state.placed_meeples)): # meeples of player p
        for m in range(len(game.state.placed_meeples[p])):
            meeple_placed = game.state.placed_meeples[p][m] # m th meeple of player p
            
            # TODO: ensure own player is in same layer regarless of p,m 
            if p == 0:  player_factor = +1 # own meeples
            else:       player_factor = -1 # opponent(s)'
            
            # extract info
            meeple_row = meeple_placed.coordinate_with_side.coordinate.row
            meeple_col = meeple_placed.coordinate_with_side.coordinate.column
            meeple_side = meeple_placed.coordinate_with_side.side.value
            
            # convert to board subgrid position and add on layer 0 (meeples)
            tile_layer = np.array(connection_region_dict[meeple_side]) * player_factor
            board_array[0,3*meeple_row:3*meeple_row+3,3*meeple_col:3*meeple_col+3] = tile_layer

def build_state_vector(game):
    for p in range(game.players):
        if p==0: # TODO: make sure AI plays as p0
            meeples_inhand_mine = game.state.meeples[p]
            score_mine = game.state.scores[p]
        else:
            meeples_inhand_opp = game.state.meeples[p]
            score_opp = game.state.scores[p]
    cards_left_in_drawpile = len(game.state.deck)
    
    state_vector = [
        cards_left_in_drawpile,
        meeples_inhand_mine,
        meeples_inhand_opp,
        score_mine,
        score_opp
    ]
    
    return state_vector

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


def interpret_board_array(board_array,state_vector):
                
    # read state vector
    cards_left, meeples_0, meeples_1, score_0, score_1 = state_vector
    
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

def estimate_potential_score(game,method='object'):
    
    # do it by meeple end-game grab
    # for some reason the copied game still impact all game objects
    if method=='object':
        game_endmove = copy.copy(game)
        PointsCollector.count_final_scores(game_endmove.state)
        scores=game_endmove.state.scores
    
    # do it via board interpreted form
    elif method=='array':
        state_vector = build_state_vector(game)

        board_array_bool = build_board_array(game,do_norm=False)
        board_array_vals = build_board_array(game,do_norm=True)
        interpret_board_dict = interpret_board_array(board_array_bool,state_vector)

        # layers 1=roads, 2=city, 3=abbeys
        point_array = np.nansum( board_array_vals[1:4], axis=0 )
        scores = []
        for p in range(2):# for 2 players
            point_array_p = point_array * interpret_board_dict[f"owned_mask_{p}"]
            scores.append(np.sum(point_array_p))
    
    return scores
    
def plot_carcassonne_board(board_array,state_vector,player_labels, interpret_board_dict, ax=None):
    # initialize plot
    if ax is None:
        fig, ax = plt.subplots()
    ax.clear()
    
    # Assuming board_array has shape (layers, height, width)
    height, width = board_array.shape[1], board_array.shape[2]

    # Start with background
    rgb_image = np.zeros((height, width, 3))
    background = 0.9
    rgb_image[:, :] = [background,background,background]
    # add each element
    rgb_image[interpret_board_dict["tile_mask"]] = [0,0.5,0]
    rgb_image[interpret_board_dict["road_mask"]] = [0.5,0.5,0.5]
    rgb_image[interpret_board_dict["intersection_mask"]] = [0.75,0.75,0.75]
    rgb_image[interpret_board_dict["city_mask"]] = [0,0,0.5]
    rgb_image[interpret_board_dict["city_shield_mask"]] = [0,0,0.75]
    rgb_image[interpret_board_dict["abbey_mask"]] = [1,0,0]
    ax.imshow(rgb_image)
    
    # Add hashed overlays using contourf hatching
    for idx, label in player_labels.items():
        key = f"owned_mask_{idx}"
        owned_mask = interpret_board_dict[key]
        color = label['color']
        if idx==0:
            hatches = ['', '----']
        else:
            hatches = ['', '||||']

        # Use contourf with hatching to overlay the owned region
        ax.contourf(
            owned_mask.astype(int),
            levels=1,
            hatches=hatches,
            alpha=0.,  # fully transparent fill
            colors=[color],  # hatch color
            zorder=5
        )
        # Set hatch color manually (defaults to black otherwise)
        #for collection in contours.collections:
        #    collection.set_edgecolor(color)
        #    collection.set_linewidth(0.5)  # optional: finer lines

                    
    # add meeples - Layer 0
    mine_y, mine_x = np.where(interpret_board_dict["meeples_mask_0"])
    oppo_y, oppo_x = np.where(interpret_board_dict["meeples_mask_1"])
    ax.plot(mine_x, mine_y, marker='x', linestyle='None', color=player_labels[0]['color'], markersize=5, markeredgewidth=2,zorder=6)
    ax.plot(oppo_x, oppo_y, marker='x', linestyle='None', color=player_labels[1]['color'], markersize=5, markeredgewidth=2,zorder=6)
    
    # add text for vector infos
    cards_left, meeples_0, meeples_1, score_0, score_1 = state_vector    # === Title (top center) ===
    ax.set_title(f"Turns Left: {cards_left}", fontsize=12, weight='bold')

    # Bottom annotation for player info
    ax.text(0.01, -0.06,
            f"[P1] {player_labels[0]['name']}   Score: {score_0}    Meeples: {meeples_0}",
            transform=ax.transAxes, fontsize=9, color=player_labels[0]['color'], family='monospace')
    ax.text(0.01, -0.12,
            f"[P2] {player_labels[1]['name']}   Score: {score_1}    Meeples: {meeples_1}",
            transform=ax.transAxes, fontsize=9, color=player_labels[1]['color'], family='monospace')
    
    # Add dashed black grid lines every 3 cells
    offset = 0.6
    for x in range(0, width, 3):
        ax.axvline(x - offset, color='black', linestyle='--', linewidth=0.5)
    for y in range(0, height, 3):
        ax.axhline(y - offset, color='black', linestyle='--', linewidth=0.5)

    ax.axis("off")
    return ax
    
def print_state(game):
    carcassonne_game_state = game.state
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

def plot_score_history(score_history,player_labels,label='Score'):
    plt.figure(figsize=(8, 4))
    plt.plot(np.array(score_history)[:,0], label=player_labels[0]['name'], color=player_labels[0]['color'], linewidth=2)
    plt.plot(np.array(score_history)[:,1], label=player_labels[1]['name'], color=player_labels[1]['color'], linewidth=2)
    plt.title(f"{label} Progression Over Game")
    plt.xlabel("Turn")
    plt.ylabel(label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # plot summary results
def plot_summary_results(final_score_history,player_labels):
    # Convert all score histories to a NumPy array for easier processing
    # Shape: (num_games, num_turns, num_players)
    # Since games may have different lengths, pad them
    max_turns = max(len(game_scores) for game_scores in final_score_history)
    num_players = len(final_score_history[0][0])

    # Initialize with NaNs for padding
    score_history_padded = np.full((len(final_score_history), max_turns, num_players), np.nan)

    for i, game_scores in enumerate(final_score_history):
        score_array = np.array(game_scores)
        score_history_padded[i, :len(score_array), :] = score_array

    # Normalize turns to percentage (0-100%)
    turns_percentage = np.linspace(0, 100, max_turns)

    # Calculate the mean score over time (across games)
    mean_score_over_time = np.nanmean(score_history_padded, axis=0)

    # Plot the mean scores over time (normalized to game %)
    plt.figure(figsize=(10, 5))
    for player_idx in range(num_players):
        plt.plot(turns_percentage, mean_score_over_time[:, player_idx], label=player_labels[player_idx]["name"],
                color=player_labels[player_idx]["color"])
    plt.xlabel("Game Progress (%)")
    plt.ylabel("Average Score")
    plt.title("Average Score Over Time (Normalized to Game %)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Extract final scores for boxplot (last score per game)
    final_scores = np.array([game_scores[-1] for game_scores in final_score_history])

    # Boxplot of final scores
    plt.figure(figsize=(6, 5))
    plt.boxplot(final_scores, labels=[player_labels[i]["name"] for i in range(num_players)])
    plt.ylabel("Final Score")
    plt.title("Final Scores Boxplot")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

def plot_final_scores_boxplot(final_score_history, player_labels=None, title="Final Score Distribution Over Games"):
    """
    Plots a boxplot of final scores for each player across multiple games.

    Args:
        final_score_history (list of list): Each inner list should contain scores of [player_1_score, player_2_score].
        player_labels (list of str, optional): Labels for the players. Default is ["Player 1", "Player 2"].
        title (str): Title of the plot.
    """
    # Default player labels
    if player_labels is None:
        player_labels = ["Player 1", "Player 2"]

    # Transpose the list to separate scores by player
    player_scores = list(zip(*final_score_history))

    # Plot setup
    plt.figure(figsize=(6, 4))
    box = plt.boxplot(
        player_scores,
        labels=player_labels,
        patch_artist=True,
        boxprops=dict(facecolor="skyblue", color="blue"),
        medianprops=dict(color="darkred"),
        whiskerprops=dict(color="blue"),
        capprops=dict(color="blue")
    )

    plt.title(title)
    plt.ylabel("Final Score")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_winrate_heatmap(win_matrix, agents):
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        100*win_matrix,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        cbar=True,
        xticklabels=agents,
        yticklabels=agents,
        annot_kws={"size": 12, "weight": "bold", "color": "black"},
        linewidths=0.5,
        linecolor='white'
    )

    ax.set_xlabel("Opponent", fontsize=14, weight="bold")
    ax.set_ylabel("Player", fontsize=14, weight="bold")
    ax.set_title("Win Rate Heatmap (percent)", fontsize=16, weight="bold")

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_scoregap_heatmap(score_gap_matrix, agents):
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        score_gap_matrix,
        annot=True,
        fmt=".1f",
        cmap="RdBu_r",  # diverging colormap: red = P0 loses, blue = P0 wins
        center=0,       # zero score gap in the middle
        xticklabels=agents,
        yticklabels=agents,
        annot_kws={"size": 12, "weight": "bold"},
        linewidths=0.5,
        linecolor='white'
    )

    ax.set_xlabel("Opponent", fontsize=14, weight="bold")
    ax.set_ylabel("Player", fontsize=14, weight="bold")
    ax.set_title("Average Score Gap (Rows vs Columns)", fontsize=16, weight="bold")

    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.show()


def compute_rewards(
    score_history,
    game_finished,
    weights=(0.01, 0.05, 0.1, 2.0)
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

def plot_scores_and_rewards(score_history, reward_history, player0_name="Player 0", player1_name="Player 1"):
    """
    Plot scores and rewards over time.

    Args:
        score_history: list of (score_p0, score_p1) at each turn
        reward_history: list of (reward_p0, reward_p1) at each turn
        player0_name: str
        player1_name: str
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # --- Scores ---
    axs[0].plot([s[0] for s in score_history], label=f"{player0_name} score", color='orange')
    axs[0].plot([s[1] for s in score_history], label=f"{player1_name} score", color='blue')
    axs[0].set_ylabel('Score')
    axs[0].set_xlabel('Turn')
    axs[0].set_title('Scores Over Time')
    axs[0].legend()
    axs[0].grid()

    # --- Rewards ---
    axs[1].plot([r[0] for r in reward_history], label=f"{player0_name} reward", color='orange')
    axs[1].plot([r[1] for r in reward_history], label=f"{player1_name} reward", color='blue')
    axs[1].set_ylabel('Reward')
    axs[1].set_xlabel('Turn')
    axs[1].set_title('Rewards Over Time')
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, window_size=10):
    """Simple moving average for smoothing."""
    x = np.array(x)
    if len(x) < window_size:
        return x  # too short
    return np.convolve(x, np.ones(window_size) / window_size, mode='valid')

def plot_training_progress(score_histories, reward_histories, player_labels, smoothing_window=5):
    """
    Plot training curves: end scores and end rewards over games.
    
    Args:
        score_histories: list of [score0, score1] per game
        reward_histories: list of [reward0, reward1] per game
        player_labels: dict with {0: {name, color}, 1: {name, color}}
        smoothing_window: int, moving average window size
    """

    N_games = len(score_histories)
    
    # Extract scores and rewards directly
    final_scores_p0 = [scores[0] for scores in score_histories]
    final_scores_p1 = [scores[1] for scores in score_histories]

    final_rewards_p0 = [rewards[0] for rewards in reward_histories]
    final_rewards_p1 = [rewards[1] for rewards in reward_histories]

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    ## --- Plot final scores ---
    axs[0].plot(final_scores_p0, label=f"{player_labels[0]['name']} score", color=player_labels[0]['color'], alpha=0.4)
    axs[0].plot(final_scores_p1, label=f"{player_labels[1]['name']} score", color=player_labels[1]['color'], alpha=0.4)

    axs[0].plot(moving_average(final_scores_p0, smoothing_window), color=player_labels[0]['color'], lw=2)
    axs[0].plot(moving_average(final_scores_p1, smoothing_window), color=player_labels[1]['color'], lw=2)

    axs[0].set_title("Final Score per Game")
    axs[0].set_xlabel("Game")
    axs[0].set_ylabel("Score")
    axs[0].legend()
    axs[0].grid(True)

    ## --- Plot final rewards ---
    axs[1].plot(final_rewards_p0, label=f"{player_labels[0]['name']} reward", color=player_labels[0]['color'], alpha=0.4)
    axs[1].plot(final_rewards_p1, label=f"{player_labels[1]['name']} reward", color=player_labels[1]['color'], alpha=0.4)

    axs[1].plot(moving_average(final_rewards_p0, smoothing_window), color=player_labels[0]['color'], lw=2)
    axs[1].plot(moving_average(final_rewards_p1, smoothing_window), color=player_labels[1]['color'], lw=2)

    axs[1].set_title("Final Reward per Game")
    axs[1].set_xlabel("Game")
    axs[1].set_ylabel("Reward")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_game_summary(player_labels,score_history,rewards_history,rewards_cumul_history,losses):
    plot_score_history(score_history, player_labels,label='Score')
    plot_score_history(rewards_history, player_labels,label='Reward')
    plot_score_history(rewards_cumul_history, player_labels,label='Cumulative Reward')
    plot_score_history(losses, player_labels,label='Loss')
