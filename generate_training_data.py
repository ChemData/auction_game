import json
from auction import *
from game_player import TrainingGames
from copy import deepcopy


t = TrainingGames('nf2')
params = {'game_state': AuctionGameState([5, 4, 3], 100, 0, 0),
          'parent': None,
          'probability': None,
          'exp_const': basic_exp_const,
          'node_dict': {0: AuctionOffer, 1: AuctionAccept},
          'is_head': True}


t.run_games(30, 100, AuctionOffer, AuctionNN, None, params, {'exp_const': '{param.__name__}'},
            explorations=200)


