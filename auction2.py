import os
import json
from copy import deepcopy
import numpy as np
import scipy.stats
import pandas as pd
from game_player import Node, basic_exp_const

"""The purpose of this over auction.py, is to manage the game state as a class
rather than as a numpy array. This should provide more flexibility and easy of
coding/debugging. It may come at the cost of speed. This should be looked into later."""


class AuctionOffer(Node):

    def new_states(self):
        """Generate and return the new states that are the result of offering different
        items for sale at different prices."""
        output = []
        for item in range(self.game_state.num_items):
            for price in self.prices[item, :]:
                output += [self.game_state.offer(item, price)]
        return output

    def _extra_calcs(self):
        """Calculate the prices offered and their probabilities."""
        num = 5
        prices = np.zeros((self.game_state.num_items, num))
        offer_output = self.child_probs.reshape((self.game_state.num_items, 3))
        for j, i in enumerate(range(-num//2+1, num//2+1)):
            prices[:, j] = offer_output[:, 1] + i*offer_output[:, 2]
        self.prices = prices.astype(int)
        probs = scipy.stats.norm.pdf(range(-num//2+1, num//2+1))
        probs = np.concatenate([probs, probs, probs]).reshape((len(offer_output), num))
        probs = probs * offer_output[:, 0].reshape((len(offer_output), 1))
        self.child_probs = probs.flatten()

    def _check_if_finished(self):
        self.winner = self.game_state.winner


class AuctionAccept(Node):

    def new_states(self):
        """Generate and return the new states that are the result of accepting/rejecting
        the deal."""
        return [self.game_state.reject(), self.game_state.accept()]

    def _check_if_finished(self):
        self.winner = self.game_state.winner


class AuctionGameState:
    """Game phases:
        0 - Making an offer.
        1 - Deciding to accept/reject an offer.
    """
    max_turns = 16
    item_payouts = [50, 20, 0]

    def __init__(self, num_items, starting_cash, active_player, phase):
        """Initialize a new game with a certain number of items.

        Args:
            num_items (list): Number of each item to start each player with.
            start_cash (int): Amount of cash to start each player with.
            active_player (int): Player who can currently make decisions.
            phase (int): Phase of the game.
        """
        self.allowed = True
        self.sellable = np.ones((2, len(num_items))) * np.array(num_items)
        self.bought = np.zeros((2, len(num_items)))
        self.offered_item = None
        self.offered_price = 0
        self.cash = [int(starting_cash)] * 2
        self.active_player = active_player
        self.phase = phase

    def __repr__(self):
        out = f'Phase {self.phase}\n'
        if self.active_player == 0:
            out += f'Player 0 (Active)\n'
        else:
            out += f'Player 0\n'
        out += f'Cash: {self.cash[0]}\n'
        out += f'Available: {self.sellable[0, :]}\n'
        out += f'Bought: {self.bought[0, :]}\n'
        if self.active_player == 1:
            out += f'\nPlayer 1 (Active)\n'
        else:
            out += f'\nPlayer 1\n'
        out += f'Cash: {self.cash[1]}\n'
        out += f'Available: {self.sellable[1, :]}\n'
        out += f'Bought: {self.bought[1, :]}\n'
        out += f'\nFor Sale: {self.offered_item} at ${self.offered_price}'

        return out

    def offer(self, item, price):
        """Have active player offer an item for sale at the specified price.

        Args:
            item (int): Number of item being offered.
            price (int): Price item is being offered for.

        Output:
            A new AuctionGameState with the updated information.
            True, if the state is allowed and False if not.
        """
        if self.phase != 0:
            raise ActionNotAllowedInPhase(f'An offer may not be made in phase {self.phase}')
        price = int(price)
        new = deepcopy(self)
        new.offered_item = item
        new.offered_price = price
        new.sellable[new.active_player, item] -= 1
        new.active_player = new.other_player

        if price > new.cash[self.active_player]:
            new.allowed = False
        if new.sellable[self.active_player, item] < 0:
            new.allowed = False

        new.phase = 1

        return new

    def accept(self):
        """Accept the current offer.
        Output:
            A new AuctionGameState with the updated information.
        """
        if self.phase != 1:
            raise ActionNotAllowedInPhase(f'An offer may not be accepted in phase {self.phase}')
        if self.offered_item is None or self.offered_price is None:
            raise BadGameState('Player cannot accept a nonexistent offer.')
        new = deepcopy(self)
        new.cash[new.active_player] -= new.offered_price
        new.cash[new.other_player] += new.offered_price
        new.bought[new.active_player, new.offered_item] += 1
        new.offered_price = 0
        new.offered_item = None

        if new.cash[new.active_player] < 0:
            new.allowed = False

        new.phase = 0

        return new

    def reject(self):
        """Reject the current offer.
        Output:
            A new AuctionGameState with the updated information.
        """
        if self.phase != 1:
            raise ActionNotAllowedInPhase(f'An offer may not be rejected in phase {self.phase}')
        if self.offered_item is None or self.offered_price is None:
            raise BadGameState('Player cannot reject a nonexistent offer.')
        new = deepcopy(self)
        new.cash[self.other_player] -= self.offered_price
        new.bought[self.other_player, self.offered_item] += 1
        new.offered_price = 0
        new.offered_item = None
        if new.cash[new.other_player] < 0:
            raise BadGameState('A player seems to have offered an item at a price above'
                               'what they can afford.')

        new.phase = 0

        return new

    @property
    def other_player(self):
        """Return the number of the nonactive player."""
        return abs(self.active_player - 1)

    def state_array(self):
        """Return an array representing the state of the game."""
        output = np.array(self.cash)
        output = np.append(output, self.sellable.flatten(), axis=0)
        output = np.append(output, self.bought.flatten(), axis=0)
        output = np.append(output, [self.offered_price], axis=0)
        offer = np.zeros(self.sellable.shape[1])
        if self.offered_item is not None:
            offer[self.offered_item] = 1
        output = np.append(output, offer, axis=0)
        return output

    @property
    def num_items(self):
        """Return the number of types of items."""
        return self.sellable.shape[1]

    @property
    def is_finished(self):
        """Return True if the current game state is the end of the game."""
        return self.bought.sum() == self.max_turns

    @property
    def winner(self):
        """Return the number of the player who won (or .5 if a tie)."""
        ranking = self.bought.sum(axis=0).argsort()[::-1]
        player0_score, player1_score = (self.bought[:, ranking]*self.item_payouts).sum(axis=1)
        if player0_score > player1_score:
            return 0
        elif player0_score < player1_score:
            return 1
        else:
            return 0.5


class FakeNN:

    def __init__(self):
        pass

    def predict(self, inputs, node_type):
        if node_type == AuctionOffer:
            return [.7, np.array([.5, 20, 3, .3, 10, 2, .2, 30, 5])]
        elif node_type == AuctionAccept:
            return [.4, np.array([.3, .7])]


class BadGameState(BaseException):
    """Raised when a player is asked to make a choice from an illegal or impossible
     game state. e.g. Tries to accept an offer if there is no offer to accept.

    This shouldn't be raised if a player makes a choice which puts the game in an illegal
    state. In that case, the state should be flagged as not allowed.
    """


class ActionNotAllowedInPhase(BaseException):
    """Raised when a player tries to take an action which is not permitted in the
    current phase. e.g. making an offer when a previous offer hasn't been resolved."""


