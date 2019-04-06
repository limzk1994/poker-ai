import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from inspect import getmembers
from random import choice
from pypokerengine.engine.poker_constants import PokerConstants
from pypokerengine.engine.card import Card
from pypokerengine.players import BasePokerPlayer
import numpy as np


class Group18Player(BasePokerPlayer):
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
    # check, raise, fold (is see needed?)
    actions_available = 4

    def __init__(self):
        # Details are included in google doc
        self.model = Sequential()

        self.model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', input_shape=(Group18Player.card_tensor_size, Group18Player.card_tensor_size, 16)))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(Group18Player.actions_available, activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])

    # feed input into neural network
    def feed_input(self, round_tensor):
        return self.model.predict(round_tensor)

    """
    Forms a 17x17 matrix of cards by forming combinations of ranks and suits and padding remaining values with zeroes.
    """
    @classmethod
    def add_card_layer(self, round_tensor, card, ctr):
        card_tensor = np.zeros((Group18Player.card_tensor_size, Group18Player.card_tensor_size))
        row_index = Group18Player.suits.index(card[0])
        col_index = Group18Player.ranks.index(card[1])
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
        pot_layer = np.zeros((Group18Player.card_tensor_size, Group18Player.card_tensor_size))
        no_of_suits = len(Group18Player.suits)
        no_of_ranks = len(Group18Player.ranks)

        if pot_amount >= Group18Player.smallest_number_of_chips * no_of_suits * no_of_ranks:
            for i in range(no_of_suits):
                np.put(pot_layer[i], list(range(no_of_ranks)), [1] * no_of_ranks)
        else:
            # number of chips in the pot in units of 10
            pot_units = pot_amount / Group18Player.smallest_number_of_chips
            #  number of columns to be filled in the 4 x 14 card tensor
            col_nums = pot_units / len(Group18Player.suits) + 1
            # number of rows to be filled in the last column of the 4 x 14 card tensor
            row_nums = pot_units % len(Group18Player.suits)
            # fill the first col_nums - 1 columns completely with ones
            for i in range(len(Group18Player.suits)):
                np.put(pot_layer[i], list(range(col_nums - 1)), [1] * (col_nums - 1))
            # fill the last column up to row_nums with ones
            for i in range(row_nums):
                np.put(pot_layer[i], [col_nums - 1], [1])
        round_tensor[ctr] = pot_layer
        return ctr + 1

    @classmethod
    def add_player_layer(self, round_tensor, ctr, is_small_blind):
        if is_small_blind:
            pot_layer = np.ones((Group18Player.card_tensor_size, Group18Player.card_tensor_size))
        else:
            pot_layer = np.zeros((Group18Player.card_tensor_size, Group18Player.card_tensor_size))
        round_tensor[ctr] = pot_layer
        ctr += 1
        return ctr

    @classmethod
    def add_betting_layer(self, round_tensor, ctr, no_of_turns_completed):
        for i in range(Group18Player.total_no_of_rounds):
            if i < no_of_turns_completed:
                round_tensor[ctr] = np.ones((Group18Player.card_tensor_size, Group18Player.card_tensor_size))
            else:
                round_tensor[ctr] = np.zeros((Group18Player.card_tensor_size, Group18Player.card_tensor_size))
            ctr += 1
        return ctr

    # TODO: return correct action and amount (currently randomly chosen)
    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state['community_card']
        no_of_layers = len(hole_card) + len(community_card) + Group18Player.total_no_of_rounds + 3
        round_tensor = np.zeros((no_of_layers, Group18Player.card_tensor_size, Group18Player.card_tensor_size))
        ctr = 0

        for card in hole_card + community_card:
            ctr = Group18Player.add_card_layer(round_tensor, card, ctr)
        ctr = Group18Player.add_extra_hand_layer(round_tensor, ctr)

        pot_amount = round_state['pot']['main']['amount']
        ctr = Group18Player.add_pot_layer(round_tensor, ctr, pot_amount)

        is_small_blind = True if round_state['big_blind_pos'] != round_state['next_player'] else False
        ctr = Group18Player.add_player_layer(round_tensor, ctr, is_small_blind)

        no_of_turns_completed = len(round_state['action_histories']) - 1
        ctr = Group18Player.add_betting_layer(round_tensor, ctr, no_of_turns_completed)

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
  return Group18Player()
