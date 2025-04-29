# module that runs a game between two agents

# first class imports
import random
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from IPython.display import Image, display
import contextlib
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
from helper_plotting import *
from agents.agents_alogirthmic import *
from agents.agents_ai import *

def two_player_game(
    board_size,
    max_turn,
    do_convert,
    do_plot,
    p0='RLAgent',
    p1='random',
    gamma=0.0,
    epsilon=0.99,
    reward_weights=(1.0, 0.0, 0.0, 1.0),
    record_rewards=True,
    do_save=True
):
    # w0=score w1=increase w2=gap w3=final turn factor
    
    # to train on algorithmics, use  and
    # score_max_own: (1.0, 0.0, 0.0, 1.0), gamma=0
    # score_potential_max_own: (1.0, 0.0, 0.0, 1.0), gamma=0.95
    # score_potential_max_gap: (0.0, 1.0, 0.0, 1.0), gamma=0.95
    # score_potential_delta_own: (0.0, 0.0, 1.0, 1.0), gamma=0.95
    # score_potential_delta_gap: (0.0, 0.5, 0.5, 1.0), gamma=0.95
    """
    Simulate a Carcassonne game between two agents, and train RL agents.

    Returns:
        score_history: list of scores over time
        rewards_history: list of (reward0, reward1) over time
    """
    starttime = time.time()
    subtile_dict = construct_subtile_dict(do_norm=True)

    game = CarcassonneGame(
        players=2,
        tile_sets=[TileSet.BASE],
        supplementary_rules=[],
        board_size=(board_size, board_size)
    )

    score_history = []
    rewards_history = []
    board_history = []
    turn_history = []
    losses = []
    
    agent_classes = {
        "random": RandomAgent(),
        "center": AgentCenter(),
        "score_max_potential_own": AgentScoreMaxPotentialOwn(),
        'human': AgentUserInput(),
        "RLAgent": RLAgent(),
    }
    player0_agent=agent_classes[p0]
    if isinstance(player0_agent,RLAgent): player0_agent.epsilon = epsilon
    player1_agent=agent_classes[p1]
    if isinstance(player1_agent,RLAgent): player1_agent.epsilon = epsilon

    player_labels = {
        0: {"name": player0_agent.name, "color": "blue"},
        1: {"name": player1_agent.name, "color": "orange"}
    }

    if do_plot:
        fig, ax = plt.subplots()
        writer = PillowWriter(fps=16)
        writer_context = writer.saving(fig, "carcassonne_game.gif", dpi=150)
    else:
        fig, ax, writer, writer_context = None, None, None, contextlib.nullcontext()

    with writer_context:
        turn = 0
        while not game.is_finished() and turn < max_turn:
            turn += 1

            player = game.get_current_player()
            valid_actions = game.get_possible_actions()

            # Select action
            if player == 0:
                action = player0_agent.select_action(valid_actions, game, player)
            else:
                action = player1_agent.select_action(valid_actions, game, player)
            
            if action is not None:
                game.step(player, action)
            
            # Save board state
            if do_convert:
                #if isinstance(action,TileAction):
                #    turn_history.append(0)
                #else:turn_history.append(1)
                turn_history.append(1)

                board_array = build_board_array(game, do_norm=True, connection_region_dict=subtile_dict)
                action_vector_array = build_state_vector(game,subtile_dict)
                action_vector_array.append(1)# this can only be trained on p0 for now
                interpret_board_dict = interpret_board_array(board_array)

                board_tensor = torch.tensor(action_vector_array, dtype=torch.float32)
                board_tensor = torch.nan_to_num(board_tensor, nan=0.0)
                board_history.append(board_tensor)

                if do_plot:
                    plot_carcassonne_board(board_array, action_vector_array, player_labels, interpret_board_dict, ax=ax)
                    writer.grab_frame()

            # Record scores
            score_history.append(game.state.scores)

            # --- Immediate reward computation ---
            rewards = compute_rewards(
                score_history,
                game_finished=False,
                weights=reward_weights
            )
            rewards_history.append(rewards)

    # compute final reward based on winner
    final_rewards = compute_rewards(score_history, game_finished=True, weights=reward_weights)
    rewards_history.append(final_rewards)
    board_history.append(board_tensor)
    turn_history.append(0) # never evaluate the end board as all the meeples are removed

    # --- After game finished: discounted training pass ---

    T = len(board_history)
    R_cumul = []
    reward_cumul = [0,0]
    for t in reversed(range(T)):
        # coompute disounted cumulative reward increase
        reward_cumul[0] = rewards_history[t][0] + gamma * reward_cumul[0]
        reward_cumul[1] = rewards_history[t][1] + gamma * reward_cumul[1]
        
        R_cumul.insert(0, reward_cumul.copy())
        
        # train for this reward step
    for t in range(T):
        board_tensor = board_history[t]
        reward_cumul = R_cumul[t]
        loss=[0,0]
        MIN_TURN_EVAL = 8 # 2 player x (meeple+tile) actions = 4 'turns' per tile, starting at tile 4 of player = tile 8 of game = turn 40
        if t>MIN_TURN_EVAL:
            if turn_history[t]==1: # i.e. Meeple turn
                for player in range(2):
                    loss_p=0
                    # train the RL agent on both player scores
                    if player==0 and isinstance(player0_agent, RLAgent):
                        #print(player,reward_cumul[player])
                        loss_p = player0_agent.train_step(board_tensor, reward_cumul[player], player)
                    if player==1 and isinstance(player1_agent, RLAgent):
                        board_tensor_inverted = board_tensor
                        board_tensor_inverted[-1]=-1
                        loss_p = player1_agent.train_step(board_tensor, reward_cumul[player], player)
                    # assign losses to each player for inspection
                    if player==0: loss[0]=loss_p
                    else: loss[1]=loss_p
        losses.append(loss)
        
    # Save model at the end
    if do_save:
        if isinstance(player0_agent, RLAgent):
            player0_agent.save_model()
        if isinstance(player1_agent, RLAgent):
            player1_agent.save_model()

    elapsed = time.time() - starttime
    print(f'Game took {elapsed:.2f} seconds')
    print(f'[RESULTS] scores: {score_history[-1]}')

    if record_rewards:
        return player_labels, score_history, rewards_history, R_cumul, losses
    else:
        return score_history

if __name__ == '__main__':

    # initial vars
    starttime = time.time()
    random.seed(0)
    
    p0="RLAgent"
    p1="random"

    # run game
    player_labels, score_history, rewards_history, rewards_cumul_history, losses = two_player_game(
        board_size=15,
        max_turn=500,
        do_convert=True,
        do_plot=False,
        p0=p0,
        p1=p1,
        record_rewards=True,
        #reward_weights=(0.1, 0.1, 0.01, 10.0),
        do_save=False # so we can tune learning rate and weights without messing up the model
    )   

    plot_game_summary(player_labels,score_history,rewards_history,rewards_cumul_history,losses)