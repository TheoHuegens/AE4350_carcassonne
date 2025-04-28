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

import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
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
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.mp1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.global_pool(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)

class RLAgent:
    def __init__(self, 
                 name="RLAgent",
                 policy_net=None,
                 alpha=0.1,
                 gamma=0.9,
                 epsilon=0.1,
                 critic_lr=0.01,
                 save_interval=1,
                 model_path="policy_net.pth"):
        
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if policy_net is None:
            self.policy_net = PolicyNet(input_channels=4).to(self.device)
        else:
            self.policy_net = policy_net.to(self.device)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=critic_lr)
        self.loss_fn = nn.MSELoss()
        self.model_path = model_path
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.save_interval = save_interval

        # Load model if it exists
        if os.path.exists(self.model_path):
            self.load_model()
            print(f"[INFO] Loaded model from {self.model_path}")

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_path)
        print(f"[INFO] Saved model to {self.model_path}")

    def load_model(self):
        self.policy_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.policy_net.eval()  # Set to evaluation mode by default when loading
    
    def select_action(self, valid_actions, game, player):
        current_board = build_board_array(game, do_norm=True)

        all_action_boards = []
        valid_indices = []
        
        for idx, action in enumerate(valid_actions):
            if action is not None:
                game_copy = copy.copy(game)
                game_copy.step(player, action)
                new_board = build_board_array(game_copy, do_norm=True)
                all_action_boards.append(torch.tensor(new_board, dtype=torch.float32))
                valid_indices.append(idx)

        if len(all_action_boards) == 0:
            return None

        board_batch = torch.stack(all_action_boards).to(self.device)
        board_batch = torch.nan_to_num(board_batch, nan=0.0)

        # --- ε-greedy strategy ---
        if random.random() < self.epsilon:
            # RANDOM action
            random_idx = random.choice(valid_indices)
            selected_action = valid_actions[random_idx]
            return selected_action
        else:
            # GREEDY action
            with torch.no_grad():
                action_scores = self.policy_net(board_batch)

            best_idx = valid_indices[torch.argmax(action_scores).item()]
            selected_action = valid_actions[best_idx]
            return selected_action

    def train_step(self, board_tensor, target_score):
        self.policy_net.train()

        board_tensor = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
        board_tensor = torch.nan_to_num(board_tensor, nan=0.0)

        target_score = torch.tensor([target_score], dtype=torch.float32).to(self.device)

        predicted_score = self.policy_net(board_tensor)

        loss = self.loss_fn(predicted_score, target_score)

        self.optimizer.zero_grad()
        
        """
        print("Target score:", target_score[0])
        print("predicted_score:", predicted_score[0])
        """
        if torch.isnan(loss):
            print("Loss is NaN, aborting step")
            return
    
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
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