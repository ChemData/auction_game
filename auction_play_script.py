import json
from auction2 import *
from copy import deepcopy

p = AuctionOffer(
    FakeNN(),
    AuctionGameState([5, 4, 3], 100, 0, 0),
    parent=None,
    probability=None,
    exp_const=basic_exp_const,
    node_dict={0: AuctionOffer, 1: AuctionAccept})

p._expand()
p2 = p.children[0]
p2._expand()
