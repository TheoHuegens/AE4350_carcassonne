# agent.py

# first class imports
import abc
import random
from typing import Any, Dict, Optional
import copy
import os
# for CNN
import torch
import torch.nn as nn
import torch.nn.functional as F
# custom library imports
from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.objects.actions.pass_action import PassAction
from wingedsheep.carcassonne.objects.actions.tile_action import TileAction
from wingedsheep.carcassonne.objects.actions.meeple_action import MeepleAction

# local imports
from helper import *
from agents.agent import Agent
from actors import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet_CNN(nn.Module):
    def __init__(self, input_channels=4):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=3)  # Now 15x15

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # -> (batch, 128, 1, 1)

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, board_batch):
        x = board_batch
        x = F.sigmoid(self.conv1(x))
        x = F.sigmoid(self.conv2(x))
        x = self.mp1(x)

        x = F.sigmoid(self.conv3(x))
        x = F.sigmoid(self.conv4(x))
        x = self.global_pool(x)

        x = x.view(x.size(0), -1)

        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)

class PolicyNet(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=21, output_dim=1):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.af1 = nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.af2 = nn.Identity()
        #self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        #self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.af3 = nn.Identity()

    def forward(self, x):
        # hidden layer
        out = self.fc1(x)
        out = self.af1(out)
        # hidden layer
        out = self.fc2(out)
        out = self.af2(out)
        # second hidden layer
        #out = self.fc3(out)
        #out = self.relu3(out)
        # output layer
        out = self.fc4(out)
        out = self.af3(out)  # output constrained to be > 0
        return out

class RLAgent:
    def __init__(self, 
                 name="RLAgent",
                 policy_net=None,
                 epsilon=0.1,
                 critic_lr=1e-12,
                 save_interval=1,
                 model_path="policy_net.pth",
                 policy_algo_init=None):
        
        self.name = name
        # put network on GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # initialise policy net
        if policy_net is None:
            self.policy_net = PolicyNet()
            if self.model_path is not None and os.path.exists(self.model_path):
                self.load_model()
            else:
                print(f"[INFO] No model found at {self.model_path}, starting fresh.")        
        else:
            self.policy_net = policy_net

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=critic_lr)
        self.loss_fn = nn.MSELoss()
        self.model_path = model_path
        self.epsilon = epsilon
        self.save_interval = save_interval
        
        self.policy_algo_init = policy_algo_init
        if policy_algo_init is not None:
            with torch.no_grad():
                # Zero all weights and biases
                for param in self.policy_net.parameters():
                    param.zero_()

                # pass all inputs to next layer as is
                self.policy_net.fc1.weight.copy_(torch.eye(self.policy_net.fc2.out_features))
                self.policy_net.fc1.bias[0] = 0.0
                # pass all inputs to next layer as is
                self.policy_net.fc2.weight.copy_(torch.eye(self.policy_net.fc2.out_features))
                self.policy_net.fc2.bias.zero_()
                # Map the final activation directly from hidden[0]
                self.policy_net.fc4.weight.zero_()
                self.policy_net.fc4.bias.zero_()
            
                
                player_vector_len = 10
                if policy_algo_init == 'score_max_own':
                    self.policy_net.fc4.weight[0][1] = 1.0
                if policy_algo_init == 'score_max_gap':
                    self.policy_net.fc4.weight[0][1] = 1.0
                    self.policy_net.fc4.weight[0][1+player_vector_len] = -1.0
                if policy_algo_init == 'score_max_potential_own':
                    self.policy_net.fc4.weight[0][1] = 1.0 # score
                    self.policy_net.fc4.weight[0][4] = 1.0 # road tiles
                    self.policy_net.fc4.weight[0][7] = 1.0 # city tiles
                    self.policy_net.fc4.weight[0][9] = 8.0 # abbeys
                if policy_algo_init == 'score_max_potential_gap':
                    self.policy_net.fc4.weight[0][1] = 1.0 # score
                    self.policy_net.fc4.weight[0][4] = 1.0 # road tiles
                    self.policy_net.fc4.weight[0][7] = 1.0 # city tiles
                    self.policy_net.fc4.weight[0][9] = 8.0 # abbeys
                    self.policy_net.fc4.weight[0][1+player_vector_len] = -1.0 # score
                    self.policy_net.fc4.weight[0][4+player_vector_len] = -1.0 # road tiles
                    self.policy_net.fc4.weight[0][7+player_vector_len] = -1.0 # city tiles
                    self.policy_net.fc4.weight[0][9+player_vector_len] = -8.0 # abbeys

            print(f"[INFO] Policy network re-initialized for {policy_algo_init} mode.")
        
        # move and inii policy net
        self.policy_net.to(self.device)
        self.policy_net.eval()

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_path)
        print(f"[INFO] Saved model to {self.model_path}")

    def load_model(self):
        self.policy_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        #self.policy_net.eval()  # Set model to evaluation mode
        print(f"[INFO] Loaded model from {self.model_path}")
    
    def select_action(self, valid_actions, game, player):
        #print(self.epsilon)
        # --- ε-greedy strategy ---
        if random.random() < self.epsilon:
            # RANDOM action
            selected_action = random.choice(valid_actions)
            # BEST HARDCODED action
            selected_action = agent_score_max_own(valid_actions,game,player)
            #selected_action = agent_score_potential_max_own(valid_actions,game,player)
            #selected_action = agent_score_potential_delta_own(valid_actions,game,player)
            #selected_action = agent_score_potential_max_gap(valid_actions,game,player)
            #selected_action = agent_score_potential_delta_gap(valid_actions,game,player)

        # --- argmax(score) policy ---
        else:
            # precompute the tile arrays
            subtile_dict = construct_subtile_dict(do_norm=True)
            all_action_boards = []
            first_action_list = [] # track action associated with this move
            
            # make action boards
            board_array_normed = build_board_array(game, do_norm=True, connection_region_dict=subtile_dict)
            board_array_bitwise = build_board_array(game, do_norm=False, connection_region_dict=subtile_dict)

            for idx, first_action in enumerate(valid_actions):
                game_copy = copy.copy(game) # needed for .get possible actions
                game_copy.step(player, first_action)
                # update action boards
                action_board_array_bitwise = update_board_array(board_array_normed,first_action,player,connection_region_dict=subtile_dict,do_norm=True)
                action_board_array_normed = update_board_array(board_array_bitwise,first_action,player,connection_region_dict=subtile_dict,do_norm=False)
                
                if isinstance(first_action, TileAction): # try all meeple placements on that tile - hard to see what's good otherwise
                    valid_second_actions = [None] # No secondary action # only look at action itself
                    #valid_second_actions = game_copy.get_possible_actions() # then look at tile-meeple action pair
                else:
                    valid_second_actions = [None] # No secondary action

                for second_action in valid_second_actions:
                    if second_action is not None: 
                        #print('we eval tile-meeple action pair')
                        # i.e. we did 1=Tile then 2=Meeple
                        game_copy_copy = copy.copy(game_copy)
                        game_copy_copy.step(player, second_action)
                        # update actions board
                        action_board_array_bitwise = update_board_array(action_board_array_bitwise,second_action,player,connection_region_dict=subtile_dict,do_norm=True)
                        action_board_array_normed = update_board_array(action_board_array_normed,second_action,player,connection_region_dict=subtile_dict,do_norm=False)
                        
                        game_state = game_copy_copy.state
                        action_game_state = [len(game_state.deck),game_state.scores[0],game_state.meeples[0],game_state.scores[1],game_state.meeples[1]]
                    # make vector after last action
                    else:
                        game_state = game_copy.state
                        action_game_state = [len(game_state.deck),game_state.scores[0],game_state.meeples[0],game_state.scores[1],game_state.meeples[1]]
                    # translate to action vector
                    action_vector_array, action_vector_dict = build_state_vector(action_game_state,action_board_array_normed,action_board_array_bitwise)

                    all_action_boards.append(torch.tensor(action_vector_array, dtype=torch.float32))
                    first_action_list.append(first_action)  # link each board to its tile action

            # create batch
            board_batch = torch.stack(all_action_boards).to(self.device)
            board_batch = torch.nan_to_num(board_batch, nan=0.0)
            # GREEDY action
            with torch.no_grad():
                action_scores = self.policy_net(board_batch)
                action_scores = action_scores.detach().cpu().numpy()

            # Find index of best board
            best_idx = np.argmax(action_scores)
            
            ### === debug === ###
            #plt.plot(action_scores)
            #plt.show()
            #print(f'best action is #{best_idx} with score {action_scores[best_idx]}')
            
            # From tile_action_list, pick the corresponding tile action
            selected_action = first_action_list[best_idx]
            
        return selected_action

    def train_step(self, board_tensor, target_score, player):
        self.policy_net.train()
        board_tensor = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
        board_tensor = torch.nan_to_num(board_tensor, nan=0.0)
            
        target_score = torch.tensor([target_score], dtype=torch.float32).to(self.device)

        predicted_score = self.policy_net(board_tensor)

        loss = self.loss_fn(predicted_score, target_score)

        self.optimizer.zero_grad()
        
        """
        print("Target score:", target_score)
        print("predicted_score:", predicted_score)
        print("predicted_score:", loss)
        """
        if torch.isnan(loss):
            print("Loss is NaN, aborting step")
            return
    
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item(), target_score.item(), predicted_score.item()
    
# === DEBUG === #
"""
# CNN encoder
input_channels = 4
conv1=nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
bn1 = nn.BatchNorm2d(16)
# m x 16x45x45
conv2=nn.Conv2d(16, 32, kernel_size=3, padding=1)
bn2 = nn.BatchNorm2d(32)
# m x 32x45x45
mp1=nn.MaxPool2d(kernel_size=3,stride=3)
# m x 32x15x15
conv3=nn.Conv2d(32, 64, kernel_size=3, padding=1)
bn3 = nn.BatchNorm2d(64)
# m x 64x15x15
conv4=nn.Conv2d(64, 128, kernel_size=3, padding=1)
bn4 = nn.BatchNorm2d(128)
# m x 128x15x15
global_pool = nn.AdaptiveAvgPool2d(1)
# m x 128x1x1
# MLP head
# m x 128
fc1 = nn.Linear(128, 256)
# m x 256
fc2 = nn.Linear(256, 128)
# m x 128
fc3 = nn.Linear(128, 1)
# m x 1

# EVALUATION
board_array = np.zeros(shape=(4,45,45))
board_tensor = torch.tensor(board_array, dtype=torch.float32)
board_batch = torch.stack([board_tensor])

x = board_batch
x = F.relu(bn1(conv1(x)))
x = F.relu(bn2(conv2(x)))
x = mp1(x)
x = F.relu(bn3(conv3(x)))
x = F.relu(bn4(conv4(x)))
x = global_pool(x)    # -> (batch_size, 128, 1, 1)
x = x.view(x.size(0), -1)   # -> (batch_size, 128)
x = F.relu(fc1(x))
x = F.relu(fc2(x))
x = fc3(x)             # -> (batch_size, 1)
x.squeeze(-1) 
"""