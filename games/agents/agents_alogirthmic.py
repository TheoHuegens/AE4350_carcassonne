# agent.py

# first class imports
import abc
import random
from typing import Any, Dict, Optional
import copy

# custom library imports
from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.objects.actions.pass_action import PassAction
from wingedsheep.carcassonne.objects.actions.tile_action import TileAction
from wingedsheep.carcassonne.objects.actions.meeple_action import MeepleAction

# local imports
from helper import *
from helper_plotting import *
from agents.agent import Agent

# === RANDOM === #
class RandomAgent(Agent):
    """A simple agent that selects random valid actions."""

    def __init__(self, name: str = "RandomAgent"):
        self.name = name

    def select_action(self, valid_actions, game, player):
        action = random.choice(valid_actions)
        return action
    
    def learn(self, state: Dict[str, Any], action: Any, reward: float, next_state: Dict[str, Any], done: bool):
        # Random agent does not learn.
        pass

# === CENTRAL === #
class AgentCenter(Agent):
    def __init__(self, name: str = "CentralAgent"):
        self.name = name

    def select_action(self, valid_actions, game, player):

        board_size = np.array(game.state.board).shape[0]
        target = [board_size / 2, board_size / 2]

        action_scores = []
        for action in valid_actions:
            if isinstance(action, PassAction):
                dist = 30**2 + 30**2  # max distance (penalize pass)
            elif isinstance(action, TileAction):
                dist = (action.coordinate.column - target[0])**2 + (action.coordinate.row - target[1])**2
            elif isinstance(action, MeepleAction):
                dist = (action.coordinate_with_side.coordinate.column - target[0])**2 + (action.coordinate_with_side.coordinate.row - target[1])**2
            else:
                dist = 30**2 + 30**2  # default penalty if unknown action
            action_scores.append(dist)

        best_action_idx = np.argmin(action_scores)
        return valid_actions[best_action_idx]

    def learn(self, *args, **kwargs):
        pass  # No learning for center-seeking agent

# === MAXIMISE OWN SCORE POTENTIAL === #
class AgentScoreMaxPotentialOwn(Agent):
    def __init__(self, name="AgentScorePotentialMaxOwn"):
        self.name = name

    def select_action(self, valid_actions, game, player):

        game_nextmove = copy.copy(game)
        score_history = []

        for action in valid_actions:
            if action is not None:
                game_nextmove = copy.copy(game)
                game_nextmove.step(player, action)

            # Collect game state
            scores = estimate_potential_score(game_nextmove)
            score_history.append(scores)

        # reward is own score
        action_scores = np.array(score_history)[:,player]

        # pick the best
        action = valid_actions[np.argmax(action_scores)]
        
        return action
    
    def learn(self, *args, **kwargs):
        # No learning for a heuristic agent
        pass

# === HUMAN INPUT === #
class AgentUserInput(Agent):
    def __init__(self, name="AgentUserInput"):
        self.name = name

    def select_action(self, valid_actions, game, player):
        player_labels = {
            0: {"name": "Player 0", "color": "orange"},
            1: {"name": "Player 1", "color": "blue"}
        }

        # show all boards resulting of each action
        n_actions = len(valid_actions)
        fig, axes = plt.subplots(n_actions, 1, figsize=(6, 4 * n_actions), squeeze=False)
        axes = axes[:, 0]

        for i, action in enumerate(valid_actions):
            ax = axes[i]

            if action is not None:
                game_copy = copy.copy(game)
                game_copy.step(player, action)

                board_array = build_board_array(game_copy, do_norm=False)
                state_vector = build_state_vector(game_copy)
                interpret_board_dict = interpret_board_array(board_array, state_vector)
                plot_carcassonne_board(board_array, state_vector, player_labels, interpret_board_dict, ax=ax)

                ax.set_title(f"Action #{i}", fontsize=12)
            else:
                ax.axis("off")
                ax.set_title(f"Action #{i} (None)", fontsize=10)

        plt.tight_layout()
        plt.show()

        # show initial state (at the end because it will scroll there by default)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
        axes = axes[0, :]  # flatten

        # show initial board
        board_array = build_board_array(game, do_norm=False)
        state_vector = build_state_vector(game)
        interpret_board_dict = interpret_board_array(board_array, state_vector)
        plot_carcassonne_board(board_array, state_vector, player_labels, interpret_board_dict, ax=axes[0])

        # show tile (if Tile phase)
        first_action = valid_actions[0]
        if isinstance(first_action, TileAction):
            tile = first_action.tile
            connecting_region_dict = construct_subtile_dict(do_norm=False)
            tile_array = build_tile_array(tile, game, 0, 0, connecting_region_dict)
            interpret_tile_dict = interpret_board_array(tile_array, state_vector)
            plot_carcassonne_board(tile_array, state_vector, player_labels, interpret_tile_dict, ax=axes[1])
            fig.suptitle(f'Place tile: {tile.description}')
        else:
            fig.suptitle(f'Place Meeple')

        plt.tight_layout()
        plt.show()

        # User input loop
        while True:
            try:
                user_choice = int(input(f"\nEnter action number (0 to {n_actions - 1}): "))
                if 0 <= user_choice < n_actions and valid_actions[user_choice] is not None:
                    return valid_actions[user_choice]
                else:
                    print("Invalid or None action. Try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def learn(self, *args, **kwargs):
        pass  # Human player doesn't learn
