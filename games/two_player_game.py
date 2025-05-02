# module that runs a game between two agents

# first class imports
import random
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from IPython.display import Image, display
import contextlib
import time
from cProfile import Profile
from pstats import SortKey, Stats
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
    do_plot,
    p0='RLAgent',
    p1='random',
    do_train=True,
    do_save=True,
    game_idx=0,
):
    """
    Simulate a Carcassonne game between two agents, and train RL agents.

    Returns:
        score_history: list of scores over time
        rewards_history: list of (reward0, reward1) over time
    """
    starttime_loop = time.time()
    
    # setup parameters for this game
    param_dict = training_plan(game_idx)
    policy_algo_init=param_dict['policy_algo_init']
    epsilon=param_dict['epsilon']
    gamma=param_dict['gamma']
    reward_weights=param_dict['reward_weights']
    learning_rate=param_dict['learning_rate']
    
    # make constants and init array
    subtile_dict = construct_subtile_dict(do_norm=True)
    
    score_history = []
    score_potential_history = []
    rewards_history = []
    board_history = []
    turn_history = []
    losses = []
    target_scores = []
    predicted_scores = []
    
    agent_classes = {
        "random": RandomAgent(),
        "center": AgentCenter(),
        "score_max_own": AgentScoreMaxOwn(),
        "score_max_gap": AgentScoreMaxGap(),
        "score_max_potential_own": AgentScoreMaxPotentialOwn(),
        "score_max_potential_gap": AgentScoreMaxPotentialGap(),
        'human': AgentUserInput(),
        #"RLAgent": RLAgent(), # special case handled later
    }
    
    # choose player agents
    if p0=="RLAgent":
        if game_idx==0: player0_agent=RLAgent(epsilon = epsilon,policy_algo_init=policy_algo_init, learning_rate=0)
        else:           player0_agent=RLAgent(epsilon = epsilon, learning_rate=learning_rate)
    else:               player0_agent=agent_classes[p0]
    if p1=="RLAgent":
        if game_idx==0: player1_agent=RLAgent(epsilon = epsilon,policy_algo_init=policy_algo_init, learning_rate=learning_rate)
        else:           player1_agent=RLAgent(epsilon = epsilon, learning_rate=learning_rate)
    else:               player1_agent=agent_classes[p1]
    
    # plotting if required
    player_labels = {
    0: {"name": p0, "color": "blue"},
    1: {"name": p1, "color": "orange"}
}

    # plots later
    if do_plot:
        fig, ax = plt.subplots()
        writer = PillowWriter(fps=16)
        writer_context = writer.saving(fig, "carcassonne_game.gif", dpi=150)
    else:
        fig, ax, writer, writer_context = None, None, None, contextlib.nullcontext()
    
    # start game
    game = CarcassonneGame(
        players=2,
        tile_sets=[TileSet.BASE],
        supplementary_rules=[],
        board_size=(board_size, board_size)
    )
    starttime_game = time.time()

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
            if do_plot or do_train or True:
                # choose whether we train on all actions or just meeple actions
                #if isinstance(action,TileAction):
                #    turn_history.append(0)
                #else:turn_history.append(1)
                turn_history.append(1)
                game_to_save = game
                board_array_normed = build_board_array(game_to_save, do_norm=True, connection_region_dict=subtile_dict)
                board_array_bitwise = build_board_array(game_to_save, do_norm=False, connection_region_dict=subtile_dict)
                game_state = [len(game_to_save.state.deck),game_to_save.state.scores[0],game_to_save.state.meeples[0],game_to_save.state.scores[1],game_to_save.state.meeples[1]]
                action_vector_array, action_vector_dict = build_state_vector(game_state,board_array_normed,board_array_bitwise)

                board_tensor = torch.tensor(action_vector_array, dtype=torch.float32)
                board_tensor = torch.nan_to_num(board_tensor, nan=0.0)
                board_history.append(board_tensor)

                if do_plot:
                    interpret_board_dict = interpret_board_array(board_array_bitwise)
                    plot_carcassonne_board(board_array_bitwise, action_vector_dict, player_labels, interpret_board_dict, ax=ax)
                    writer.grab_frame()

            # Record scores
            scores = game.state.scores
            scores_potential = estimate_potential_score(game,method='object') # only use the object method here as it needs a deepocy and takes long
            #scores = game.state.scores
            score_history.append(scores)
            score_potential_history.append(scores_potential)

            # --- Immediate reward computation ---
            rewards = compute_rewards(
                score_history,score_potential_history,
                game_finished=False,
                weights=reward_weights
            )
            rewards_history.append(rewards)
    
    if do_plot: display(Image(filename="carcassonne_game.gif"))
    starttime_postproc = time.time()

    # compute final reward based on winner
    if do_train:
        final_rewards = compute_rewards(score_history,score_potential_history, game_finished=True, weights=reward_weights)
        rewards_history.append(final_rewards)
        board_history.append(board_tensor)
        turn_history.append(0) # never evaluate the end board as all the meeples are removed

        # --- After game finished: discounted training pass ---

        T = len(board_history)
        rewards_cumul_history = []
        reward_cumul = [0,0]
        for t in reversed(range(T)):
            # coompute disounted cumulative reward increase
            reward_cumul[0] = rewards_history[t][0] + gamma * reward_cumul[0]
            reward_cumul[1] = rewards_history[t][1] + gamma * reward_cumul[1]
            
            rewards_cumul_history.insert(0, reward_cumul.copy())
            
            # train for this reward step
        for t in range(T):
            board_tensor = board_history[t]
            reward_cumul = rewards_cumul_history[t]
            loss=[0,0]
            target_score=[0,0]
            predicted_score=[0,0]

            MIN_TURN_EVAL = 20 # 2 player x (meeple+tile) actions = 4 'turns' per tile, starting at tile 4 of player = tile 8 of game = turn 40
            if t>MIN_TURN_EVAL:
                if turn_history[t]==1: # i.e. Meeple turn
                    for player in range(2):
                        loss_p=0
                        # train the RL agent on both player scores
                        if player==0 and isinstance(player0_agent, RLAgent):
                            # player 0
                            loss[0], target_score[0], predicted_score[0] = player0_agent.train_step(board_tensor, reward_cumul[0], 0)
                            # adjust tensor and train with player 1 data too
                            board_tensor_inverted = swap_halves_tensor(board_tensor)                            
                            loss[1], target_score[1], predicted_score[1]  = player0_agent.train_step(board_tensor_inverted, reward_cumul[1], 1)
            losses.append(loss)
            target_scores.append(target_score)
            predicted_scores.append(predicted_score)

    # Save model at the end
    if do_save:
        if isinstance(player0_agent, RLAgent):
            player0_agent.save_model()
    
    # plot neural network for this game's training
    if do_plot and do_train:
        if isinstance(player0_agent, RLAgent):
            fig_nn,ax_nn = plot_network(player0_agent.policy_net,input_data=action_vector_dict,game_idx=game_idx)
        plot_game_summary(player_labels,score_history,rewards_history,rewards_cumul_history,losses,target_scores,predicted_scores)

    #print(f'Setup took    {starttime_loop - starttime_game:.2f} seconds')
    #print(f'Game took     {starttime_postproc - starttime_game:.2f} seconds')
    #print(f'Wrapping took {time.time() - starttime_postproc:.2f} seconds')
    print(f'[RESULTS] time {time.time() - starttime_loop:.2f} seconds')
    print(f'[RESULTS] scores: {score_history[-1]}')

    if do_train:
        return player_labels, score_history, rewards_history, rewards_cumul_history, losses, target_scores, predicted_scores, player0_agent, player1_agent, action_vector_dict
    else:
        return score_history, None, None, None, None, None, None, None, None, None

if __name__ == '__main__':

    # initial vars
    random.seed(0)
    # run game
    do_profile = False

    # and profile what takes long
    if do_profile:
        with Profile() as profile:
            player_labels, score_history, rewards_history, rewards_cumul_history, losses, target_scores, predicted_scores, agent0, agent1, state_vector = two_player_game(
                board_size=15,
                max_turn=500,
                do_plot=True,
                p0="RLAgent", # RLAgent
                p1="random",
                do_train=False,
                do_save=False, # so we can tune learning rate and weights without messing up the model
                game_idx=1 # set no >0 to use settings from training plan
            )           
        Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats()

    else:
        player_labels, score_history, rewards_history, rewards_cumul_history, losses, target_scores, predicted_scores, agent0, agent1, state_vector = two_player_game(
            board_size=15,
            max_turn=500,
            do_plot=True,
            p0="RLAgent", # RLAgent
            p1="score_max_potential_own",
            do_train=True,
            do_save=False, # so we can tune learning rate and weights without messing up the model
            game_idx=1 # set no >0 to use settings from training plan
        )
