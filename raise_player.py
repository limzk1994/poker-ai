from inspect import getmembers
from random import choice
from pypokerengine.engine.poker_constants import PokerConstants
from pypokerengine.engine.card import Card
from pypokerengine.players import BasePokerPlayer
import numpy as np


class RaisedPlayer(BasePokerPlayer):
    # number of poker rounds
    total_no_of_rounds = len(filter(lambda x: type(x[1]) == int, getmembers(PokerConstants.Street))) - 1
    # array of cards ranks obtained from PyPokerEngine
    ranks = [Card.RANK_MAP[key] for key in Card.RANK_MAP]
    # array of card suits obtained from PyPokerEngine
    suits = [Card.SUIT_MAP[key] for key in Card.SUIT_MAP]
    # size of card_tensor
    card_tensor_size = len(ranks) + len(suits)
    # number of chips in the poker pot is a multiple of this unit
    smallest_number_of_chips = 10

    """
    Forms a 17x17 matrix of cards by forming combinations of ranks and suits and padding remaining values with zeroes.
    """
    @classmethod
    def add_card_layer(self, round_tensor, card, ctr):
        card_tensor = np.zeros((RaisedPlayer.card_tensor_size, RaisedPlayer.card_tensor_size))
        row_index = RaisedPlayer.suits.index(card[0])
        col_index = RaisedPlayer.ranks.index(card[1])
        np.put(card_tensor[row_index], [col_index], [1])
        round_tensor[ctr] = card_tensor
        return ctr + 1

    @classmethod
    def add_extra_hand_layer(self, round_tensor, ctr):
        hand_layer = round_tensor[0] + round_tensor[1]
        for i in range(2, ctr):
            hand_layer += round_tensor[i]
        round_tensor[ctr] = hand_layer
        return ctr + 1

    @classmethod
    def add_pot_layer(self, round_tensor, ctr, pot_amount):
        pot_layer = np.zeros((RaisedPlayer.card_tensor_size, RaisedPlayer.card_tensor_size))
        if pot_amount >= RaisedPlayer.smallest_number_of_chips * len(RaisedPlayer.ranks) * len(RaisedPlayer.suits):
            for i in range(len(RaisedPlayer.suits)):
                no_of_ranks = len(RaisedPlayer.ranks)
                np.put(pot_layer[i], list(range(no_of_ranks)), [1] * no_of_ranks)
        else:
            # number of chips in the pot in units of 10
            pot_units = pot_amount / RaisedPlayer.smallest_number_of_chips
            #  number of columns to be filled in the 3 x 14 card tensor
            col_nums = pot_units / len(RaisedPlayer.suits) + 1
            # number of rows to be filled in the last column of the 3 x 14 card tensor
            row_nums = pot_units % len(RaisedPlayer.suits)
            # fill the first col_nums - 1 columns completely with ones
            for i in range(len(RaisedPlayer.suits)):
                np.put(pot_layer[i], list(range(col_nums - 1)), [1] * (col_nums - 1))
            # fill the last column up to row_nums with ones
            for i in range(row_nums):
                np.put(pot_layer[i], [col_nums - 1], [1])
        round_tensor[ctr] = pot_layer
        return ctr + 1

    @classmethod
    def add_player_layer(self, round_tensor, ctr, is_small_blind):
        if is_small_blind:
            pot_layer = np.ones((RaisedPlayer.card_tensor_size, RaisedPlayer.card_tensor_size))
        else:
            pot_layer = np.zeros((RaisedPlayer.card_tensor_size, RaisedPlayer.card_tensor_size))
        round_tensor[ctr] = pot_layer
        ctr += 1
        return ctr

    @classmethod
    def add_betting_layer(self, round_tensor, ctr, no_of_turns_completed):
        for i in range(no_of_turns_completed):
            round_tensor[ctr] = np.ones((RaisedPlayer.card_tensor_size, RaisedPlayer.card_tensor_size))
            ctr += 1
        for i in range(no_of_turns_completed, RaisedPlayer.total_no_of_rounds):
            round_tensor[ctr] = np.zeros((RaisedPlayer.card_tensor_size, RaisedPlayer.card_tensor_size))
            ctr += 1
        return ctr

    # TODO: return correct action and amount (currently randomly chosen)
    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state['community_card']
        no_of_layers = len(hole_card) + len(community_card) + RaisedPlayer.total_no_of_rounds + 3
        round_tensor = np.zeros((no_of_layers, RaisedPlayer.card_tensor_size, RaisedPlayer.card_tensor_size))
        ctr = 0

        for card in hole_card + community_card:
            ctr = RaisedPlayer.add_card_layer(round_tensor, card, ctr)
        ctr = RaisedPlayer.add_extra_hand_layer(round_tensor, ctr)

        pot_amount = round_state['pot']['main']['amount']
        ctr = RaisedPlayer.add_pot_layer(round_tensor, ctr, pot_amount)

        is_small_blind = True if round_state['big_blind_pos'] != round_state['next_player'] else False
        ctr = RaisedPlayer.add_player_layer(round_tensor, ctr, is_small_blind)

        no_of_turns_completed = len(round_state['action_histories']) - 1
        ctr = RaisedPlayer.add_betting_layer(round_tensor, ctr, no_of_turns_completed)

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