# agent.py

# first class imports
import abc
import random
from typing import Any, Dict, Optional
import copy

# local imports
from helper import *

class Agent(abc.ABC):
    """Abstract base class for any Carcassonne RL agent."""

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def select_action(self, state: Dict[str, Any]) -> Any:
        """Select an action based on the current state."""
        pass

    @abc.abstractmethod
    def learn(self, state: Dict[str, Any], action: Any, reward: float, next_state: Dict[str, Any], done: bool):
        """Learn from the transition."""
        pass

    def save(self, path: str) -> None:
        """Optional: Save the agent's learned parameters."""
        pass

    def load(self, path: str) -> None:
        """Optional: Load the agent's learned parameters."""
        pass

