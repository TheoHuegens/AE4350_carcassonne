# module that runs a pytest game between two agents

# first class imports
import random
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from IPython.display import Image, display
import contextlib
import pytest
import time
# custom library imports
from wingedsheep.carcassonne.carcassonne_game import CarcassonneGame
from wingedsheep.carcassonne.carcassonne_game_state import CarcassonneGameState, GamePhase
from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.objects.actions.pass_action import PassAction
from wingedsheep.carcassonne.objects.actions.tile_action import TileAction
from wingedsheep.carcassonne.objects.actions.meeple_action import MeepleAction
from wingedsheep.carcassonne.objects.meeple_type import MeepleType
from wingedsheep.carcassonne.tile_sets.supplementary_rules import SupplementaryRule
from wingedsheep.carcassonne.tile_sets.tile_sets import TileSet
# local imports
from helper import *
from agents.agent import Agent
from agents.agents_alogirthmic import *
from agents.agents_ai import *
from actors import *
from two_player_game import two_player_game

@pytest.mark.profile
def test_profile_two_player_game():
    # Settings for a fast, small game
    board_size = 15
    max_turn = 500
    do_convert = True
    do_plot = False

    player0_agent = RandomAgent()
    player1_agent = RLAgent()

    # Timing buckets
    timings = {
        "select_action": 0.0,
        "game_step": 0.0,
        "convert_board": 0.0,
        "plot": 0.0,
        "train_step": 0.0,
        "reward_compute": 0.0,
        "other": 0.0
    }

    # Start game
    start_total = time.perf_counter()

    game = CarcassonneGame(
        players=2,
        tile_sets=[TileSet.BASE],
        supplementary_rules=[],
        board_size=(board_size, board_size)
    )

    turn = 0
    while not game.is_finished() and turn < max_turn:
        turn += 1

        player = game.get_current_player()
        valid_actions = game.get_possible_actions()

        # --- select_action ---
        start = time.perf_counter()
        if player == 0:
            action = player0_agent.select_action(valid_actions, game, player)
        else:
            action = player1_agent.select_action(valid_actions, game, player)
        timings["select_action"] += time.perf_counter() - start

        # --- game_step ---
        if action is not None:
            start = time.perf_counter()
            game.step(player, action)
            timings["game_step"] += time.perf_counter() - start

        # --- board convert ---
        if do_convert:
            start = time.perf_counter()
            board_array = build_board_array(game, do_norm=True)
            board_tensor = torch.tensor(board_array, dtype=torch.float32)
            timings["convert_board"] += time.perf_counter() - start

        # --- training (immediate reward) ---
        start = time.perf_counter()
        rewards = compute_rewards(
            [game.state.scores],
            game_finished=False,
            weights=(0.1, 1.0, 0.5, 10.0)
        )
        timings["reward_compute"] += time.perf_counter() - start

        start = time.perf_counter()
        if isinstance(player0_agent, RLAgent):
            player0_agent.train_step(board_tensor, target_score=rewards[0])
        if isinstance(player1_agent, RLAgent):
            player1_agent.train_step(board_tensor, target_score=rewards[1])
        timings["train_step"] += time.perf_counter() - start

        # (No plot in this fast test)

    total_elapsed = time.perf_counter() - start_total

    # --- Print timing summary ---
    print("\n=== Profiling Summary ===")
    for key, t in timings.items():
        print(f"{key:>15}: {t:.4f} sec")
    print(f"{'Total time':>15}: {total_elapsed:.4f} sec")
    print("==========================\n")

    # Assertions (optional): game should finish within reasonable time
    assert total_elapsed < 60, "Profiling test took too long!"

if __name__=='__main__':
    test_profile_two_player_game()