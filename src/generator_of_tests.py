import random

from hold_em import count_win_probabilities
from hold_em import get_full_deck


NUM_TESTS_5 = 100
NUM_TESTS_4 = 55
NUM_TESTS_3 = 19
NUM_TESTS_0 = 4


def generate_test_group(*,
                        num_tests: int,
                        num_community_cards: int,
                        test_data_file_name: str,
                        tests_group_name: str) -> None:

    with open(test_data_file_name, 'w', encoding='utf-8') as out:
        print(f'{tests_group_name} = [', file=out)

        for _ in range(num_tests):
            full_deck = get_full_deck()

            players_private_cards = []
            num_players = random.randint(2, 5)
            for _ in range(num_players):
                first_card = random.choice(list(full_deck))
                full_deck.remove(first_card)
                second_card = random.choice(list(full_deck))
                full_deck.remove(second_card)
                players_private_cards.append((first_card.text_repr(), second_card.text_repr()))

            known_community_cards = []
            for _ in range(num_community_cards):
                community_card = random.choice(list(full_deck))
                full_deck.remove(community_card)
                known_community_cards.append(community_card.text_repr())

            num_already_dropped_cards = random.randint(15, 25)
            already_dropped_cards = []
            for _ in range(num_already_dropped_cards):
                already_dropped_card = random.choice(list(full_deck))
                full_deck.remove(already_dropped_card)
                already_dropped_cards.append(already_dropped_card.text_repr())

            right_win_probabilities = count_win_probabilities(
                players_private_cards=players_private_cards,
                known_community_cards=known_community_cards,
                already_dropped_cards=already_dropped_cards
            )

            print(
                f'''\
    [
        {players_private_cards},
        {known_community_cards},
        {already_dropped_cards},
        {right_win_probabilities}
    ],
                ''',
                file=out)

        print(']', file=out)


if __name__ == '__main__':
    generate_test_group(num_tests=NUM_TESTS_5,
                        num_community_cards=5,
                        test_data_file_name="five_community_cards_tests.py",
                        tests_group_name="FIVE_COMMUNITY_CARDS_TESTS")
    generate_test_group(num_tests=NUM_TESTS_4,
                        num_community_cards=4,
                        test_data_file_name="four_community_cards_tests.py",
                        tests_group_name="FOUR_COMMUNITY_CARDS_TESTS")
    generate_test_group(num_tests=NUM_TESTS_3,
                        num_community_cards=3,
                        test_data_file_name="three_community_cards_tests.py",
                        tests_group_name="THREE_COMMUNITY_CARDS_TESTS")
    generate_test_group(num_tests=NUM_TESTS_0,
                        num_community_cards=0,
                        test_data_file_name="zero_community_cards_tests.py",
                        tests_group_name="ZERO_COMMUNITY_CARDS_TESTS")
