import unittest
from auction import *


class TestAuctionGameState(unittest.TestCase):

    def setUp(self):
        self.state = AuctionGameState([5, 4, 3], 100, 0)

    def test_offer(self):
        self.assertEqual(self.state.sellable[0, 1], 4)
        new = self.state.offer(1, 20)
        self.assertEqual(new.sellable[0, 1], 3)
        self.assertEqual(new.active_player, 1)
        self.assertEqual(new.offered_item, 1)
        self.assertEqual(new.offered_price, 20)
        self.assertTrue(new.allowed)

    def test_offer_too_little_money(self):
        new = self.state.offer(1, 200)
        self.assertFalse(new.allowed)

    def test_offer_no_item(self):
        self.state.sellable[0, 1] = 0
        new = self.state.offer(1, 20)
        self.assertFalse(new.allowed)

    def test_accept(self):
        self.assertEqual(self.state.sellable[0, 1], 4)
        new = self.state.offer(1, 20)
        new2 = new.accept()
        self.assertCountEqual(new2.cash, [120, 80])
        self.assertEqual(new2.bought[1, 1], 1)
        self.assertEqual(new2.sellable[0, 1], 3)
        self.assertTrue(new2.allowed)

    def test_accept_too_little_money(self):
        self.state.cash[1] = 40
        new = self.state.offer(1, 50)
        new2 = new.accept()
        self.assertFalse(new2.allowed)

    def test_reject(self):
        self.assertEqual(self.state.sellable[0, 1], 4)
        new = self.state.offer(1, 20)
        new2 = new.reject()
        self.assertCountEqual(new2.cash, [80, 100])
        self.assertEqual(new2.bought[0, 1], 1)
        self.assertEqual(new2.sellable[0, 1], 3)
        self.assertTrue(new2.allowed)

