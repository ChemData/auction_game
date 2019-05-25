import os
import json
import numpy as np
import scipy.stats
import pandas as pd


def game_start(num_items, starting_cash):
    """Initialize a new game with a certain number of items."""
    output = np.zeros((2, 1+2*len(num_items)))
    output[:, 0] = starting_cash
    output[:, 1:1+len(num_items)] = num_items
    return output


def allowed_offers(game_state, **extra_vars):
    """Determine what offers are permitted and return those resultant board states."""
    output = []



def price_spacing(offer_output):
    """Return the prices and probabilities to offer each item for sale at."""
    num = 5
    prices = np.zeros((len(offer_output), num))
    for j, i in enumerate(range(-num//2+1, num//2+1)):
        prices[:, j] = offer_output[:, 1] + i*offer_output[:, 2]
    prices = prices.astype(int)
    probs = scipy.stats.norm.pdf(range(-num//2+1, num//2+1))
    probs = np.concatenate([probs, probs, probs]).reshape((len(offer_output), num))
    return {'prices': prices, 'probs': probs}


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


d = np.array([[.4, 20, 3], [.3, 10, 2], [.3, 8, 3]])
z = price_spacing(d)

