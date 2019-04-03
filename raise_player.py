from keras import backend as K
from random import randint, choice
from pypokerengine.engine.card import Card
from pypokerengine.players import BasePokerPlayer
import numpy as np

class RaisedPlayer(BasePokerPlayer):
    # array of cards ranks obtained from PyPokerEngine
    ranks = [Card.RANK_MAP[key] for key in Card.RANK_MAP]
    # array of card suits obtained from PyPokerEngine
    suits = [Card.SUIT_MAP[key] for key in Card.SUIT_MAP]
    # size of card_tensor
    card_tensor_size = len(ranks) + len(suits)
    # han
    hand_size = 5

    """
    Forms a 17x17 matrix of cards by forming combinations of ranks and suits and padding remaining values with zeroes.
    """
    @classmethod
    def get_card_tensor(self, card):
        card_tensor = np.zeros((RaisedPlayer.card_tensor_size, RaisedPlayer.card_tensor_size))
        row_index = RaisedPlayer.suits.index(card[0])
        col_index = RaisedPlayer.ranks.index(card[1])
        card_tensor[row_index][col_index] = 1
        return card_tensor

    # TODO: return correct action and amount (currently randomly chosen)
    def declare_action(self, valid_actions, hole_card, round_state):
        hand_tensor = np.zeros((RaisedPlayer.hand_size, RaisedPlayer.card_tensor_size, RaisedPlayer.card_tensor_size))
        ctr = 0
        for card in hole_card:
            hand_tensor[ctr] = RaisedPlayer.get_card_tensor(card)
            ctr += 1

        # placeholder for the actual action to be taken by the agent
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