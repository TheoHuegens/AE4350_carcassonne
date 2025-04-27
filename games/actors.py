import random
import numpy as np
from typing import Optional
import copy
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from wingedsheep.carcassonne.carcassonne_game import CarcassonneGame
from wingedsheep.carcassonne.carcassonne_game_state import CarcassonneGameState, GamePhase
from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.objects.actions.pass_action import PassAction
from wingedsheep.carcassonne.objects.actions.tile_action import TileAction
from wingedsheep.carcassonne.objects.actions.meeple_action import MeepleAction
from wingedsheep.carcassonne.objects.meeple_type import MeepleType
from wingedsheep.carcassonne.tile_sets.supplementary_rules import SupplementaryRule
from wingedsheep.carcassonne.tile_sets.tile_sets import TileSet

from helper import *

def agent_random(valid_actions,game,player=0):
    # this plays at random
    
    action: Optional[Action] = random.choice(valid_actions)
    return action

def agent_center(valid_actions,game,player=0):
    # this plays closest to the center
    board_size = np.array(game.state.board).shape[0]
    target = [board_size/2,board_size/2]
    
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

def agent_score_max_own(valid_actions,game,player=0):
    
    game_nextmove = copy.copy(game)
    score_history = []

    for action in valid_actions:
        if action is not None:
            game_nextmove.step(player, action)

        # Collect game state
        board_array = build_board_array(game_nextmove)
        state_vector = build_state_vector(game_nextmove)
        score_history.append([state_vector[3], state_vector[4]])

    score_mine = np.array(score_history)[:,player]
    action_index = np.argmax(score_mine)
    action = valid_actions[action_index]
    
    return action

def agent_score_potential_max_own(valid_actions,game,player=0):
    
    current_scores = estimate_potential_score(game)
    score_history = []

    for action in valid_actions:
        if action is not None:
            game_nextmove = copy.copy(game)
            game_nextmove.step(player, action)

        # Collect game state
        scores = estimate_potential_score(game_nextmove)

        score_history.append(scores)
    
    p_me = player
    p_opp = player+1
    if p_opp>=game.state.players: p_opp=0
    
    # maximmise own score
    action_scores = np.array(score_history)[:,p_me]

    # pick the best
    action_index = np.argmax(action_scores)
    action = valid_actions[action_index]
    
    return action

def agent_score_potential_max_gap(valid_actions,game,player=0):
    
    current_scores = estimate_potential_score(game)
    score_history = []

    for action in valid_actions:
        if action is not None:
            game_nextmove = copy.copy(game)
            game_nextmove.step(player, action)

        # Collect game state
        scores = estimate_potential_score(game_nextmove)

        score_history.append(scores)
    
    p_me = player
    p_opp = player+1
    if p_opp>=game.state.players: p_opp=0
    
    # maximise score gap
    action_scores = np.array(score_history)[:,p_me] - np.array(score_history)[:,p_opp]

    # pick the best
    action_index = np.argmax(action_scores)
    action = valid_actions[action_index]
    
    return action

def agent_score_potential_delta_own(valid_actions,game,player=0):
    
    current_scores = estimate_potential_score(game)
    score_history = []

    for action in valid_actions:
        if action is not None:
            game_nextmove = copy.copy(game)
            game_nextmove.step(player, action)

        # Collect game state
        scores = estimate_potential_score(game_nextmove)

        score_history.append(scores)
    
    p_me = player
    p_opp = player+1
    if p_opp>=game.state.players: p_opp=0
    
    # maxmimise own score increase
    action_scores = np.array(score_history)[:,p_me] - current_scores[p_me]

    # pick the best
    action_index = np.argmax(action_scores)
    action = valid_actions[action_index]
    
    return action

def agent_score_potential_delta_gap(valid_actions,game,player=0):
    
    current_scores = estimate_potential_score(game)
    score_history = []

    for action in valid_actions:
        if action is not None:
            game_nextmove = copy.copy(game)
            game_nextmove.step(player, action)

        # Collect game state
        scores = estimate_potential_score(game_nextmove)

        score_history.append(scores)
    
    p_me = player
    p_opp = player+1
    if p_opp>=game.state.players: p_opp=0

    # maxmimise score gap increase
    action_scores = (np.array(score_history)[:,p_me] - np.array(score_history)[:,p_opp]) - (current_scores[p_me]-current_scores[p_opp])

    # pick the best
    action_index = np.argmax(action_scores)
    action = valid_actions[action_index]
    
    return action

def agent_user_input(valid_actions, game, player=0):
    """
    Shows each valid action result in a vertical stack of plots,
    highlighting placed tiles and meeples. Asks user to choose.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
    axes = axes[0, :]  # flatten

    # show initial board
    player_labels = {
        0: {"name": "Player 0", "color": "orange"},
        1: {"name": "Player 1", "color": "blue"}
    }
    board_array = build_board_array(game, do_norm=False)
    state_vector = build_state_vector(game)
    interpret_board_dict = interpret_board_array(board_array, state_vector)
    plot_carcassonne_board(board_array,state_vector,player_labels,interpret_board_dict,ax=axes[0])
    
    # show tile (if it is Tile phase)
    first_action = valid_actions[0]
    if isinstance(first_action,TileAction):
        tile = first_action.tile        
        connecting_region_dict = construct_subtile_dict(do_norm=False)
        tile_array = build_tile_array(tile,game,0,0,connecting_region_dict)
        interpret_tile_dict = interpret_board_array(tile_array, state_vector)
        plot_carcassonne_board(tile_array,state_vector,player_labels,interpret_tile_dict,ax=axes[1])
        fig.suptitle(f'Place tile: {tile.description}')
    else:
        fig.suptitle(f'Place Meeple')
    plt.tight_layout()
    plt.show()

    # show all possible boards resulting from action
    n_actions = len(valid_actions)
    fig, axes = plt.subplots(n_actions, 1, figsize=(6, 4 * n_actions), squeeze=False)
    axes = axes[:, 0]  # flatten

    for i, action in enumerate(valid_actions):
        ax = axes[i]

        if action is not None:
            # Simulate the game state after the action
            game_copy = copy.copy(game)
            game_copy.step(player, action)

            # Build and interpret board
            board_array = build_board_array(game_copy, do_norm=False)
            state_vector = build_state_vector(game_copy)
            interpret_board_dict = interpret_board_array(board_array, state_vector)
            plot_carcassonne_board(board_array,state_vector,player_labels,interpret_board_dict,ax=ax)
            """
            # Highlight the tile placed
            if hasattr(action, "pos") or isinstance(action, (tuple, list)):
                pos = action.pos if hasattr(action, "pos") else action[0]
                x, y = pos[1], pos[0]
                rect = mpatches.Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

            # Highlight placed meeple if present
            if hasattr(action, "meeple") and action.meeple is not None:
                meeple_pos = action.pos  # Usually same as tile pos
                mx, my = meeple_pos[1], meeple_pos[0]
                ax.plot(mx, my, 'o', markersize=12, markeredgewidth=2, markeredgecolor='red', markerfacecolor='none')
            """
            ax.set_title(f"Action #{i}", fontsize=12)
        else:
            ax.axis("off")
            ax.set_title(f"Action #{i} (None)", fontsize=10)
            
    plt.tight_layout()
    plt.show()

    # Prompt user to choose
    while True:
        try:
            user_choice = int(input(f"\nEnter action number (0 to {n_actions - 1}): "))
            if 0 <= user_choice < n_actions and valid_actions[user_choice] is not None:
                return valid_actions[user_choice]
            else:
                print("Invalid or None action. Try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
