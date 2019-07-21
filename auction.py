import os
import json
from copy import deepcopy
import numpy as np
import scipy.stats
import pandas as pd
import keras as ks
from game_player import Node, basic_exp_const, BasicNeuralNetwork

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
        num_items = self.game_state.num_items
        prices = np.zeros((num_items, num))
        item_probs = self.child_probs[:num_items].reshape((-1, 1))
        offer_output = self.child_probs[self.game_state.num_items:].reshape((num_items, 2))
        # If, for any item, one offer is greater than the cash of the active player,
        # a new set of offers is created with a mean of cash/2 and an std of std/8
        cash = self.game_state.cash[self.game_state.active_player]
        for i, row in enumerate(offer_output):
            if row[0] + abs(row[1])*num//2 > 1:
                offer_output[i, 0] = 1/2
                offer_output[i, 1] = 1/8

        for j, i in enumerate(range(-num//2+1, num//2+1)):
            prices[:, j] = (offer_output[:, 0] + i*abs(offer_output[:, 1]))*cash
        self.prices = prices.astype(int)
        probs = scipy.stats.norm.pdf(range(-num//2+1, num//2+1))
        probs = np.concatenate([probs, probs, probs]).reshape((len(offer_output), num))
        probs = probs * item_probs
        self.child_probs = probs.flatten()

    def _target_data(self, winner):
        """Return the target (i.e. , y) data from this node based on who won."""
        z = self.pi(1)
        out = z.reshape(self.game_state.num_items, -1).sum(1)
        out = np.concatenate([out, self.weighted_avg_and_std(self.prices, z).flatten()])
        return np.append(self._win_value(winner), out)

    @staticmethod
    def weighted_avg_and_std(values, weights):
        """
        Return the weighted average and standard deviation.

        """
        weights = weights.reshape(values.shape).copy()
        # Catch any row where all values are 0 and replace with 1
        weights[np.where(weights.sum(axis=1) == 0), :] = 1

        average = np.average(values, weights=weights, axis=1)
        variance = np.average((values.T - average).T ** 2, weights=weights, axis=1)
        return np.array([average, np.sqrt(variance)]).T


class AuctionAccept(Node):

    def new_states(self):
        """Generate and return the new states that are the result of accepting/rejecting
        the deal."""
        return [self.game_state.reject(), self.game_state.accept()]


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
        if -1*price > new.cash[self.active_player]:
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
        new.cash[self.other_player] -= min(0, self.offered_price)
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

    @property
    def state_array(self):
        """Return an array representing the state of the game."""
        output = np.array(self.cash)
        if self.active_player == 1:
            output = np.append(output, self._reverse(self.sellable).flatten(), axis=0)
            output = np.append(output, self._reverse(self.bought).flatten(), axis=0)
        else:
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
        if not self.is_finished:
            return None
        ranking = self.bought.sum(axis=0).argsort()[::-1]
        player0_score, player1_score = (self.bought[:, ranking]*self.item_payouts).sum(axis=1)
        if player0_score > player1_score:
            return 0
        elif player0_score < player1_score:
            return 1
        else:
            return 0.5

    def _reverse(self, array):
        """Reverses an data array so that the values for player 0 and player 1 are
        reversed."""
        return array[::-1, :]

    @staticmethod
    def starting_state_generator(**kwargs):
        """Returns a function which will return freshly initialized game states with the
        desired first player."""
        def gen_func(first_player):
            return AuctionGameState(**kwargs, active_player=first_player)
        return gen_func


class FakeNN:

    def __init__(self):
        pass

    def predict(self, inputs, node_type):
        if node_type == AuctionOffer:
            return [.7, np.array([.5, .3, .2, 20, 3, 10, 2, 30, 5])]
        elif node_type == AuctionAccept:
            return [.4, np.array([.3, .7])]

    @property
    def code(self):
        return 1


class AuctionNN(BasicNeuralNetwork):
    loss_names = {0: ['win_prob_loss', 'item_choice_loss', 'distributions_loss'],
                  1: ['win_prob_loss', 'move_choice_loss']}
    output_divisions = {0: (1, 4), 1: (1,)}

    def _create_new(self, save=False):

        self.models = [self._offer_model(), self._accept_model()]
        self.ids = [0, 0]
        if save:
            for i, model in enumerate(self.models):
                os.makedirs(os.path.join(self.model_folder, f'model {i}'), exist_ok=True)
                model.save(os.path.join(self.model_folder, f'model {i}', '0.hdf5'))
                new_info = pd.DataFrame([[0, 0, 'random start', i]],
                                        columns=['set', 'epoch', 'base_model_id', 'model_type'])
                self._add_model_info(new_info, i)

    def _offer_model(self):
        inp = ks.Input((18,))
        x = ks.layers.Dense(50,
                            activation='relu',
                            kernel_initializer='random_uniform',
                            bias_initializer='zeros')(inp)
        x = ks.layers.Dense(50,
                            activation='relu',
                            kernel_initializer='random_uniform',
                            bias_initializer='zeros')(x)
        out1 = ks.layers.Dense(1,
                               activation='relu',
                               kernel_initializer='random_uniform',
                               bias_initializer='zeros',
                               name='win_prob')(x)
        out2 = ks.layers.Dense(3,
                               activation='softmax',
                               kernel_initializer='random_uniform',
                               bias_initializer='zeros',
                               name='item_choice')(x)
        out3 = ks.layers.Dense(6,
                               activation='linear',
                               kernel_initializer='random_uniform',
                               bias_initializer='zeros',
                               name='distributions')(x)

        model = ks.Model(inputs=inp, outputs=[out1, out2, out3])
        model.compile('RMSprop', ['binary_crossentropy', 'categorical_crossentropy',
                                  'mean_squared_error'])
        return model

    def _accept_model(self):
        inp = ks.Input((18,))
        x = ks.layers.Dense(50,
                            activation='relu',
                            kernel_initializer='random_uniform',
                            bias_initializer='zeros')(inp)
        x = ks.layers.Dense(50,
                            activation='relu',
                            kernel_initializer='random_uniform',
                            bias_initializer='zeros')(x)
        out1 = ks.layers.Dense(1,
                               activation='relu',
                               kernel_initializer='random_uniform',
                               bias_initializer='zeros',
                               name='win_prob')(x)
        out2 = ks.layers.Dense(2,
                               activation='softmax',
                               kernel_initializer='random_uniform',
                               bias_initializer='zeros',
                               name='move_choice')(x)

        model = ks.Model(inputs=inp, outputs=[out1, out2])
        model.compile('RMSprop', ['binary_crossentropy', 'categorical_crossentropy'])
        return model

    def predict(self, inputs, node_type):
        inputs = inputs.reshape((1, -1))
        if node_type == AuctionOffer:
            output = self.models[0].predict(inputs)
        else:
            output = self.models[1].predict(inputs)
        return [output[0][0, 0], np.concatenate([x.flatten() for x in output[1:]])]


class BadGameState(BaseException):
    """Raised when a player is asked to make a choice from an illegal or impossible
     game state. e.g. Tries to accept an offer if there is no offer to accept.

    This shouldn't be raised if a player makes a choice which puts the game in an illegal
    state. In that case, the state should be flagged as not allowed.
    """


class ActionNotAllowedInPhase(BaseException):
    """Raised when a player tries to take an action which is not permitted in the
    current phase. e.g. making an offer when a previous offer hasn't been resolved."""


