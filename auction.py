import os
import json
import numpy as np
import pandas as pd


def game_start(num_items, starting_cash):
    """Initialize a new game with a certain number of items."""
    output = np.zeros((2, 1+2*len(num_items)))
    output[:, 0] = starting_cash
    output[:, 1:1+len(num_items)] = num_items
    return output


def allowed_offers(game_state):
    """Determine what offers are permitted and return those resultant board states."""


class Node:

    def __init__(self, player, model, game_state, play_state, parent):
        self.player = player
        self.model = model
        self.game_state = game_state
        self.play_state = play_state
        self.parent = parent
        self.children = []

    def _expand(self):
        """Expand the children of this node."""
        pass


with open('auction_params.json', 'r') as fp:
    p = json.load(fp)


p = game_start([5,4,3], 100)
