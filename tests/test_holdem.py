"""
Testing function for texas hold'em win probabilities counter
"""

import pytest

from src import count_win_probabilities

from five_community_cards_tests import FIVE_COMMUNITY_CARDS_TESTS
from four_community_cards_tests import FOUR_COMMUNITY_CARDS_TESTS
from three_community_cards_tests import THREE_COMMUNITY_CARDS_TESTS
from zero_community_cards_tests import ZERO_COMMUNITY_CARDS_TESTS
from custom_tests import CUSTOM_TESTS


EPS = 0.0001


@pytest.mark.parametrize(
    'players_private_cards, known_community_cards, already_dropped_cards, right_win_probabilities',
    (
        []
        + CUSTOM_TESTS
        + FIVE_COMMUNITY_CARDS_TESTS
        + FOUR_COMMUNITY_CARDS_TESTS
        + THREE_COMMUNITY_CARDS_TESTS
        + ZERO_COMMUNITY_CARDS_TESTS
    ),
    ids=str)
def test_count_win_probabilities(players_private_cards,
                                 known_community_cards,
                                 already_dropped_cards,
                                 right_win_probabilities):
    win_probabilities = count_win_probabilities(players_private_cards=players_private_cards,
                                                known_community_cards=known_community_cards,
                                                already_dropped_cards=already_dropped_cards)
    assert len(win_probabilities) == len(right_win_probabilities)
    for i in range(len(right_win_probabilities)):
        assert -EPS < win_probabilities[i] - right_win_probabilities[i] < EPS, (
            f'Player #{i}: {win_probabilities[i]} instead of {right_win_probabilities[i]}'
        )
