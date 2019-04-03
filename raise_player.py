from keras import backend as K
from inspect import getmembers
from pprint import pprint
from random import randint, choice
from pypokerengine.engine.poker_constants import PokerConstants
from pypokerengine.engine.card import Card
from pypokerengine.players import BasePokerPlayer
import numpy as np

class RaisedPlayer(BasePokerPlayer):
    no_of_rounds = getmembers(PokerConstants)
    # array of cards ranks obtained from PyPokerEngine
    ranks = [Card.RANK_MAP[key] for key in Card.RANK_MAP]
    # array of card suits obtained from PyPokerEngine
    suits = [Card.SUIT_MAP[key] for key in Card.SUIT_MAP]
    # size of card_tensor
    card_tensor_size = len(ranks) + len(suits)

    """
    Forms a 17x17 matrix of cards by forming combinations of ranks and suits and padding remaining values with zeroes.
    """
    @classmethod
    def add_card_layer(self, round_tensor, card, ctr):
        card_tensor = np.zeros((RaisedPlayer.card_tensor_size, RaisedPlayer.card_tensor_size))
        row_index = RaisedPlayer.suits.index(card[0])
        col_index = RaisedPlayer.ranks.index(card[1])
        card_tensor[row_index][col_index] = 1
        round_tensor[ctr] = card_tensor
        return ctr + 1

    @classmethod
    def add_extra_hand_layer(self, round_tensor, ctr):
        hand_layer = round_tensor[0] + round_tensor[1]
        for i in range(ctr):
            hand_layer += round_tensor[i]
        round_tensor[ctr] = hand_layer
        return ctr + 1

    @classmethod
    def add_player_layer(self, round_tensor, ctr):
        return ctr

    @classmethod
    def add_round_layer(self, round_tensor, ctr, no_of_turns_completed):
        for i in range(no_of_turns_completed):
            round_tensor[ctr] = np.ones((RaisedPlayer.card_tensor_size, RaisedPlayer.card_tensor_size))
            ctr += 1
        round_tensor[ctr] = np.zeros((RaisedPlayer.card_tensor_size, RaisedPlayer.card_tensor_size))
        ctr += 1
        return ctr

    # TODO: return correct action and amount (currently randomly chosen)
    def declare_action(self, valid_actions, hole_card, round_state):
        pprint(round_state)
        community_card = round_state['community_card']
        no_of_layers = (len(hole_card) + len(community_card) + 1) * 3
        round_tensor = np.zeros((no_of_layers, RaisedPlayer.card_tensor_size, RaisedPlayer.card_tensor_size))
        ctr = 0
        for card in hole_card + community_card:
            ctr = RaisedPlayer.add_card_layer(round_tensor, card, ctr)
        ctr = RaisedPlayer.add_extra_hand_layer(round_tensor, ctr)

        ctr = RaisedPlayer.add_player_layer(round_tensor, ctr)
        no_of_turns_completed = len(round_state['action_histories']) - 1
        ctr = RaisedPlayer.add_round_layer(round_tensor, no_of_turns_completed, ctr)
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