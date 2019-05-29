import os
import json
import numpy as np
import scipy.stats
import pandas as pd


class Node:

    def __init__(self, model, game_state, parent, probability, exp_const, node_dict={}):
        """
        Args:
            model (placholder): The neural network used for decision making.
            game_state (GameState): Class that stores information about the state of the
                game.
            parent (Node): The parent node of this node.
            probability (float): Probability of moving to this node.
            exp_const (func): A function which returns the exploration constant to use.
            node_dict (dict): Dictionary that defines what Node class is used for each
                game state.
        """

        self.model = model
        self.game_state = game_state
        self.parent = parent
        self.exp_const = exp_const
        self.node_dict = node_dict
        self.children = []
        self.child_states = np.array([])
        self.child_probs = np.array([])
        self.winner = None
        self._check_if_finished()

        self.p = probability
        self.N = 0
        self.W = 0
        self.Q = 0

        try:
            self.depth = self.parent.depth + 1
        except AttributeError:
            self.depth = 0

    def _find_leaf(self):
        """Find a leaf from this node."""
        if self.winner is not None:
            return self
        if len(self.children) == 0:
            return self
        return self._child_to_explore().find_leaf()

    def _child_to_explore(self):
        """Determine which child should be explored to find a leaf."""
        child_vals = self.child_explore_vals
        options = np.where(child_vals == child_vals.max())[0]
        return self.children[np.random.choice(options)]

    @property
    def child_explore_vals(self):
        """Value of choosing its children in an MCTS leaf exploration."""
        if len(self.children) == 0:
            raise ValueError('This node has no child_explore_vals because it has no children')
        qs = np.array([c.Q for c in self.children])
        ps = np.array([c.p for c in self.children])
        ns = np.array([c.N for c in self.children])

        return qs + self.exp_const(self.depth) * ps * np.sqrt(ns.sum())/(1 + ns)

    def _expand(self):
        """Expand the children of this node."""
        self.v, self.child_probs = self.model.predict(self.game_state.state_array, self.__class__)
        self._extra_calcs()
        self.child_states = self.new_states()
        self._cull_disallowed_states()
        self._child_prob_normalization()
        for i, state in enumerate(self.child_states):
            type_of_node = self.node_dict.get(state.phase)
            self.children += [type_of_node(self.model, state, self, self.child_probs[i],
                                           self.exp_const, self.node_dict)]

    def _extra_calcs(self):
        """Perform any follow up calculations using the NN output to compute child
        probabilities."""
        pass

    def _cull_disallowed_states(self):
        """Set the probability of any state which is not allowed to 0."""
        for i, state in enumerate(self.child_states):
            if not state.allowed:
                self.child_probs[i] = 0

    def _child_prob_normalization(self):
        """Normalize the probabilities of choosing the different children so that the
        total is 1."""
        self.child_probs /= self.child_probs.sum()

    def new_states(self):
        """Determine the new board states for each child."""
        return []

    def _check_if_finished(self):
        """Determine if the game is completed and who the winner (if any) is."""
        pass


def basic_exp_const(depth):
    return 1.212

