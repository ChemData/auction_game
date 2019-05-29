import os
import json
import numpy as np
import scipy.stats
import pandas as pd
from game_player import Node


class AuctionOffer(Node):

    @staticmethod
    def game_start(num_items, starting_cash):
        """Initialize a new game with a certain number of items."""
        output = np.zeros((2, 2+3*len(num_items)))
        output[:, 0] = starting_cash
        output[:, 1:1+len(num_items)] = num_items
        return output

    def allowed_offers(self, game_state, extra_vars):
        """Determine what offers are permitted and return those resultant board states."""
        prices = extra_vars['prices']
        probs = extra_vars['probs']
        output = []
        for item in range(prices.shape[0]):
            for price in range(prices.shape[1]):
                new_state = game_state.copy()
                # Check for enough cash to sell
                if prices[item, price] > game_state[0, 0]:
                    probs[item, price] = 0
                elif game_state[0, 1+item] < 1:
                    probs[item, price] = 0
                else:
                    # Add an offer
                    new_state[0, 1+item+2*prices.shape[0]] = 1
                    new_state[0, 1+3*prices.shape[0]] = prices[item, price]
                    # Remove an item
                    new_state[0, 1+item] -= 1
                output += [new_state]

        output = np.array(output)
        output = output.reshape((prices.shape[0], prices.shape[1],
                                 game_state.shape[0], game_state.shape[1]))

        # Normalize probs
        probs = probs/probs.sum()

        return output, probs

    def price_spacing(self, offer_output):
        """Return the prices and probabilities to offer each item for sale at."""
        num = 5
        prices = np.zeros((len(offer_output), num))
        for j, i in enumerate(range(-num//2+1, num//2+1)):
            prices[:, j] = offer_output[:, 1] + i*offer_output[:, 2]
        prices = prices.astype(int)
        probs = scipy.stats.norm.pdf(range(-num//2+1, num//2+1))
        probs = np.concatenate([probs, probs, probs]).reshape((len(offer_output), num))
        probs = probs * offer_output[:, 0].reshape((len(offer_output), 1))
        return {'prices': prices, 'probs': probs}


class AuctionAccept(Node):

    def allowed_acceptors(self, game_state, extra_vars):
        """Return the new board states for the outcomes of accepting/rejecting offers."""
        probs = extra_vars['prob']
        output = []
        offered = np.where(game_state[1, 7:10] == 1)[0, 0]
        yes_state = game_state[1, offered+1]




