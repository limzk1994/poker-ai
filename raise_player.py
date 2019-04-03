from keras import backend as K
from random import randint, choice
from pypokerengine.engine.card import Card
from pypokerengine.players import BasePokerPlayer
import numpy as np

class RaisedPlayer(BasePokerPlayer):
    card_matrix = np.empty((1, 1))

    """
    Forms a 17x17 matrix of cards by forming combinations of ranks and suits and padding remaining values with zeroes.
    """
    @classmethod
    def init_card_matrix(self):
        # array of cards ranks obtained from PyPokerEngine
        ranks = [Card.RANK_MAP[key] for key in Card.RANK_MAP]
        # array of card suits obtained from PyPokerEngine
        suits = [Card.SUIT_MAP[key] for key in Card.SUIT_MAP]
        matrix_size = len(ranks) + len(suits)
        RaisedPlayer.card_matrix = np.chararray((matrix_size, matrix_size), itemsize=2)
        # padding matrix with 0s
        RaisedPlayer.card_matrix.fill('0')
        for i in range(len(suits)):
            cards = [suits[i] + rank for rank in ranks]
            rank_indices = list(range(len(ranks)))
            np.put(RaisedPlayer.card_matrix[i], rank_indices, cards)

    # TODO: return correct action and amount (currently randomly chosen)
    def declare_action(self, valid_actions, hole_card, round_state):
        RaisedPlayer.init_card_matrix()
        action_dict = choice(valid_actions)

        action = action_dict['action']
        return action

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
      pass

    def receive_street_start_message(self, street, round_state):
      pass

    def receive_game_update_message(self, action, round_state):
      pass

    def receive_round_result_message(self, winners, hand_info, round_state):
      pass


def setup_ai():
  return RaisedPlayer()