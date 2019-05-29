import unittest
import numpy as np
from auction import *


class TestAuctionOffer(unittest.TestCase):

    def setUp(self):
        self.node = AuctionOffer(0, 0, 0, 0, 0)

    def test_game_start(self):
        start = self.node.game_start([5, 4, 3], 100)
        correct = np.array(
            [[100.,   5.,   4.,   3.,   0.,   0.,   0.,   0.,   0.,   0., 0],
             [100.,   5.,   4.,   3.,   0.,   0.,   0.,   0.,   0.,   0., 0]])
        np.testing.assert_array_almost_equal(start, correct)

    def test_allowed_offers_too_little_money(self):
        """Make sure that options which are too expensive are assigned a probability of 0.
        """
        state = self.node.game_start([5, 4, 3], 20)
        d = np.array([[.4, 20, 3], [.5, 10, 2], [.1, 8, 3]])
        z = self.node.price_spacing(d)
        probs = self.node.allowed_offers(state, z)[1]
        correct = np.array([[0.02475284, 0.11093455, 0.18290016, 0, 0.],
         [0.03094106, 0.13866819, 0.2286252, 0.13866819, 0.03094106],
        [0.00618821, 0.02773364, 0.04572504, 0.02773364, 0.00618821]])
        np.testing.assert_array_almost_equal(probs, correct)
        self.assertEqual(probs.sum(), 1)  # Ensure that probabilities sum to 1

    def test_allowed_offers_item_lacking(self):
        """Make sure that options where there is no item available for sale are assigned
        a probability of 0."""
        state = self.node.game_start([5, 4, 0], 100)
        d = np.array([[.4, 20, 3], [.5, 10, 2], [.1, 8, 3]])
        z = self.node.price_spacing(d)
        probs = self.node.allowed_offers(state, z)[1]
        correct = np.array([[0.02421719, 0.10853393, 0.1789422, 0.10853393, 0.02421719],
                            [0.03027149, 0.13566741, 0.22367775, 0.13566741, 0.03027149],
                            [0, 0, 0, 0, 0]])
        np.testing.assert_array_almost_equal(probs, correct)
        self.assertEqual(probs.sum(), 1)  # Ensure that probabilities sum to 1

