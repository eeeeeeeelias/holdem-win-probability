"""
Win probability counter for texas hold'em.
"""
from collections import Counter
import itertools
from math import factorial
from multiprocessing import Pool
import typing as tp


NUM_CHUNKS = 32


SUIT_LETTER_TO_VALUE = {
    'S': 0,
    'C': 1,
    'D': 2,
    'H': 3,
}


SUIT_VALUE_TO_LETTER = {
    0: 'S',
    1: 'C',
    2: 'D',
    3: 'H',
}


SUIT_VALUE_TO_SYMBOL = {
    0: '♠',
    1: '♣',
    2: '♦',
    3: '♥',
}


RANK_VALUE_TO_SYMBOL = {
    11: 'J',
    12: 'Q',
    13: 'K',
    14: 'A',
}

RANK_SYMBOL_TO_VALUE = {
    'T': 10,
    'J': 11,
    'Q': 12,
    'K': 13,
    'A': 14,
}

MIN_RANK = 2
MAX_RANK = 14
MIN_SUIT = 0
MAX_SUIT = 3


NUM_COMMUNITY_CARDS = 5
NUM_PRIVATE_CARDS = 2
NUM_HAND_CARDS = 5


class Card:
    def __init__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs:
            raw_rank = args[0][:-1]
            rank = int(RANK_SYMBOL_TO_VALUE.get(raw_rank, raw_rank))
            assert MIN_RANK <= rank <= MAX_RANK, "There are ranks from 2 to A (14)"
            raw_suit = args[0][-1]
            assert raw_suit in SUIT_LETTER_TO_VALUE, "There are 4 suits: S, C, D, H"
            suit = SUIT_LETTER_TO_VALUE[raw_suit]
        elif kwargs and not args:
            rank = kwargs['rank']
            suit = kwargs['suit']
        else:
            raise TypeError('Create card via either Card(rank=9, suit=0) or Card(\'9S\')')
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return (
            f'{RANK_VALUE_TO_SYMBOL.get(self.rank, self.rank)}'
            f'{SUIT_VALUE_TO_SYMBOL[self.suit]}'
        )

    def text_repr(self):
        return (
            f'{RANK_VALUE_TO_SYMBOL.get(self.rank, self.rank)}'
            f'{SUIT_VALUE_TO_LETTER.get(self.suit)}'
        )

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))

    def __lt__(self, other):
        if self.rank == other.rank:
            return self.suit < other.suit
        return self.rank < other.rank


def get_ranks_frequencies(cards: list[Card]) -> list[tuple[int, int]]:
    ranks_frequencies = Counter([card.rank for card in cards]).most_common()
    ranks_frequencies.sort(key=lambda item: (-item[1], -item[0]))
    return ranks_frequencies


def get_suits_frequencies(cards: list[Card]) -> list[tuple[int, int]]:
    suits_frequencies = Counter([card.suit for card in cards]).most_common()
    suits_frequencies.sort(key=lambda item: (-item[1], -item[0]))
    return suits_frequencies


def get_most_frequent_suit_cards_ranks(cards: list[Card]) -> list[int]:
    suits_frequencies = get_suits_frequencies(cards)
    return list({card.rank for card in cards if card.suit == suits_frequencies[0][0]})


def _look_ranks_for_straight(ranks: list[int]) -> tp.Optional[list[int]]:
    ranks.sort(reverse=True)
    if len(ranks) < NUM_HAND_CARDS:
        return None
    if MAX_RANK in ranks:
        ranks.append(1)
    for i in range(len(ranks) - NUM_HAND_CARDS + 1):
        max_straight_rank = ranks[i]
        min_straight_rank = ranks[i + NUM_HAND_CARDS - 1]
        if max_straight_rank - min_straight_rank == NUM_HAND_CARDS - 1:
            return [max_straight_rank]
        continue
    return None


def look_for_royal_flush(available_cards: list[Card]) -> tp.Optional[list[int]]:
    one_suit_cards_ranks = get_most_frequent_suit_cards_ranks(available_cards)
    possible_straight = _look_ranks_for_straight(one_suit_cards_ranks)
    if possible_straight is None:
        return None
    if possible_straight[0] != MAX_RANK:
        return None
    return possible_straight


def look_for_straight_flush(available_cards: list[Card]) -> tp.Optional[list[int]]:
    one_suit_cards_ranks = get_most_frequent_suit_cards_ranks(available_cards)
    return _look_ranks_for_straight(one_suit_cards_ranks)


def look_for_four_of_a_kind(available_cards: list[Card]) -> tp.Optional[list[int]]:
    ranks_frequencies = get_ranks_frequencies(available_cards)
    if ranks_frequencies[0][1] < 4:
        return None
    return [
        ranks_frequencies[0][0],
        max(rank for rank, _ in ranks_frequencies[1:])
    ]


def look_for_full_house(available_cards: list[Card]) -> tp.Optional[list[int]]:
    ranks_frequencies = get_ranks_frequencies(available_cards)
    if ranks_frequencies[0][1] < 3 or ranks_frequencies[1][1] < 2:
        return None
    return [
        ranks_frequencies[0][0],
        ranks_frequencies[1][0]
    ]


def look_for_flush(available_cards: list[Card]) -> tp.Optional[list[int]]:
    suits_frequencies = get_suits_frequencies(available_cards)
    if suits_frequencies[0][1] < NUM_HAND_CARDS:
        return None
    one_suit_cards_ranks = [
        card.rank
        for card in available_cards
        if card.suit == suits_frequencies[0][0]
    ]
    one_suit_cards_ranks.sort(reverse=True)
    return one_suit_cards_ranks[:NUM_HAND_CARDS]


def look_for_straight(available_cards: list[Card]) -> tp.Optional[list[int]]:
    ranks = list({card.rank for card in available_cards})
    return _look_ranks_for_straight(ranks)


def look_for_three_of_a_kind(available_cards: list[Card]) -> tp.Optional[list[int]]:
    ranks_frequencies = get_ranks_frequencies(available_cards)
    if ranks_frequencies[0][1] < 3:
        return None
    return [rank for rank, _ in ranks_frequencies[:NUM_HAND_CARDS - 2]]


def look_for_two_pair(available_cards: list[Card]) -> tp.Optional[list[int]]:
    ranks_frequencies = get_ranks_frequencies(available_cards)
    if ranks_frequencies[0][1] < 2 or ranks_frequencies[1][1] < 2:
        return None
    first_combination_rank = ranks_frequencies[0][0]
    second_combination_rank = ranks_frequencies[1][0]
    ranks_frequencies = sorted(ranks_frequencies[2:], key=lambda item: -item[0])
    first_high_card = ranks_frequencies[0][0]
    return [first_combination_rank, second_combination_rank, first_high_card]


def look_for_pair(available_cards: list[Card]) -> tp.Optional[list[int]]:
    ranks_frequencies = get_ranks_frequencies(available_cards)
    if ranks_frequencies[0][1] < 2:
        return None
    return [rank for rank, _ in ranks_frequencies[:NUM_HAND_CARDS - 1]]


def look_for_high_card(available_cards: list[Card]) -> tp.Optional[list[int]]:
    ranks_frequencies = get_ranks_frequencies(available_cards)
    return [rank for rank, _ in ranks_frequencies[:NUM_HAND_CARDS]]


class Combination:
    def __init__(self, *, name: str, rank: int, num_important_cards_ranks: int, looker):
        self.name = name
        self.rank = rank
        self.num_important_cards_ranks = num_important_cards_ranks
        self.looker = looker

    def __repr__(self):
        return f'#{self.rank}: {self.name}'


COMBINATIONS = [
    Combination(
        name='royal flush', rank=0, num_important_cards_ranks=1, looker=look_for_royal_flush),
    Combination(
        name='straight flush', rank=1, num_important_cards_ranks=1, looker=look_for_straight_flush),
    Combination(
        name='four of a kind', rank=2, num_important_cards_ranks=2, looker=look_for_four_of_a_kind),
    Combination(
        name='full house', rank=3, num_important_cards_ranks=2, looker=look_for_full_house),
    Combination(
        name='flush', rank=4, num_important_cards_ranks=5, looker=look_for_flush),
    Combination(
        name='straight', rank=5, num_important_cards_ranks=1, looker=look_for_straight),
    Combination(
        name='three of a kind', rank=6, num_important_cards_ranks=3,
        looker=look_for_three_of_a_kind),
    Combination(
        name='two pair', rank=7, num_important_cards_ranks=3, looker=look_for_two_pair),
    Combination(
        name='pair', rank=8, num_important_cards_ranks=4, looker=look_for_pair),
    Combination(
        name='high card', rank=9, num_important_cards_ranks=5, looker=look_for_high_card)
]


class Hand:
    def __init__(self, combination: Combination, important_cards_ranks: list[int]):
        assert combination.num_important_cards_ranks == len(important_cards_ranks)
        self.combination = combination
        self.important_cards_ranks = important_cards_ranks

    # 'less' == 'worse'
    def __lt__(self, other: 'Hand'):
        if self.combination.rank == other.combination.rank:
            return self.important_cards_ranks < other.important_cards_ranks
        return self.combination.rank > other.combination.rank

    def __eq__(self, other: 'Hand'):
        return (
            self.combination.rank == other.combination.rank
            and self.important_cards_ranks == other.important_cards_ranks
        )

    def __repr__(self):
        return f'combination={self.combination}, important_cards_ranks={self.important_cards_ranks}'


def get_best_hand(available_cards: list[Card]) -> Hand:
    assert len(available_cards) == NUM_COMMUNITY_CARDS + NUM_PRIVATE_CARDS
    for possible_combination in COMBINATIONS:
        important_card_ranks: tp.Optional[list[int]] = possible_combination.looker(available_cards)
        if important_card_ranks is None:
            continue
        assert len(important_card_ranks) == possible_combination.num_important_cards_ranks, (
            f'{possible_combination.name}: {available_cards} -> {important_card_ranks}'
        )
        return Hand(combination=possible_combination, important_cards_ranks=important_card_ranks)


def get_winners_ids_by_hands(players_hands: list[Hand]) -> list[int]:
    best_hand: Hand = max(players_hands)
    return [i for i, hand in enumerate(players_hands) if hand == best_hand]


def get_winners_ids(new_community_cards, known_community_cards, players_private_cards) -> list[int]:
    all_community_cards = list(known_community_cards) + list(new_community_cards)
    players_hands = [
        get_best_hand(list(player_cards) + list(all_community_cards))
        for player_cards
        in players_private_cards
    ]
    return get_winners_ids_by_hands(players_hands)


def count_chunk_win_outcomes(one_chunk_data):
    players_private_cards = one_chunk_data['players_private_cards']
    new_community_cards_list = one_chunk_data['new_community_cards_list']
    chunk_start = one_chunk_data['chunk_start']
    chunk_stop = one_chunk_data['chunk_stop']
    known_community_cards = one_chunk_data['known_community_cards']

    players_win_outcomes = [0 for _ in players_private_cards]

    for index in range(chunk_start, chunk_stop):
        winners_ids = get_winners_ids(new_community_cards=new_community_cards_list[index],
                                      known_community_cards=known_community_cards,
                                      players_private_cards=players_private_cards)
        for winner_id in winners_ids:
            players_win_outcomes[winner_id] += 1.0 / len(winners_ids)

    return players_win_outcomes


def get_full_deck() -> tp.Set[Card]:
    return {
        Card(rank=rank, suit=suit)
        for rank, suit
        in itertools.product(range(MIN_RANK, MAX_RANK + 1), range(MIN_SUIT, MAX_SUIT + 1))
    }


def get_outcomes_number(num_community_card_to_complete: int, deck: tp.Set[Card]) -> int:
    total_outcomes: int = 1
    for i in range(num_community_card_to_complete):
        total_outcomes *= len(deck) - i
    total_outcomes //= factorial(num_community_card_to_complete)
    return total_outcomes


def count_win_probabilities(players_private_cards: list[tuple[str, str]],
                            known_community_cards: list[str],
                            already_dropped_cards: list[str] = None):
    deck: tp.Set[Card] = get_full_deck()

    if already_dropped_cards:
        already_dropped_cards = {Card(raw_card) for raw_card in already_dropped_cards}
    else:
        already_dropped_cards = set()
    known_community_cards = {Card(raw_card) for raw_card in known_community_cards}

    players_private_cards = [
        (Card(raw_first_card), Card(raw_second_card))
        for raw_first_card, raw_second_card
        in players_private_cards
    ]

    deck -= already_dropped_cards
    deck -= known_community_cards
    for player_cards in players_private_cards:
        for card in player_cards:
            deck.remove(card)

    num_community_card_to_complete = NUM_COMMUNITY_CARDS - len(known_community_cards)

    players_win_outcomes = [0 for _ in players_private_cards]

    total_outcomes: int = get_outcomes_number(num_community_card_to_complete, deck)

    if len(known_community_cards) > 0:
        for outcome in itertools.combinations(deck, num_community_card_to_complete):
            winners_ids = get_winners_ids(new_community_cards=outcome,
                                          known_community_cards=known_community_cards,
                                          players_private_cards=players_private_cards)
            for winner_id in winners_ids:
                players_win_outcomes[winner_id] += 1.0 / len(winners_ids)
    else:
        new_community_cards_list = list(itertools.combinations(deck, num_community_card_to_complete))

        chunk_size = (total_outcomes + NUM_CHUNKS - 1) // NUM_CHUNKS

        data_chunked = []
        for chunk_id in range(NUM_CHUNKS):
            data_chunked.append(
                {
                    'chunk_start': chunk_id * chunk_size,
                    'chunk_stop': min((chunk_id + 1) * chunk_size, len(new_community_cards_list)),
                    'players_private_cards': players_private_cards,
                    'new_community_cards_list': new_community_cards_list,
                    'known_community_cards': known_community_cards,
                }
            )

        pool = Pool(NUM_CHUNKS)
        win_outcomes_chunked = pool.map(count_chunk_win_outcomes, data_chunked)
        for win_outcomes in win_outcomes_chunked:
            for i in range(len(players_win_outcomes)):
                players_win_outcomes[i] += win_outcomes[i]

    return [num_win_outcomes / total_outcomes for num_win_outcomes in players_win_outcomes]


# How to use.
if __name__ == '__main__':
    # [0.2558139534883721, 0.636766334440753, 0.10741971207087486]
    probabilities = count_win_probabilities(
        players_private_cards=[('2C', 'QC'), ('KH', 'KS'), ('AD', '8D')],
        known_community_cards=['3C', '6H', 'QS'],
        already_dropped_cards=[]
    )
    print(probabilities)
    # [0.15982639238453195, 0.5475531471486976, 0.2926204604667699]
    probabilities = count_win_probabilities(
        players_private_cards=[('2C', 'QC'), ('KH', 'KS'), ('AD', '8D')],
        known_community_cards=[],
        already_dropped_cards=[]
    )
    print(probabilities)
