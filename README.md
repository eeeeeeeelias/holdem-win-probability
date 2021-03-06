# holdem-win-probability

![Python >= 3.6](https://img.shields.io/badge/python->=3.6-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![cov: 96%](https://img.shields.io/badge/codecov-96%25-brightgreen)

A Python package to count win probability in [Texas Hold'em](https://en.wikipedia.org/wiki/Texas_hold_%27em).

## Installation
```
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps holdem-win-probability-eeeeeeeelias
```

## Usage
```python
from holdem_probability import count_win_probabilities

# [1.0, 0.0] (royal flush after flop, 1st player wins)
probabilities = count_win_probabilities(
    players_private_cards=[('AH', 'KH'), ('AD', '8D')],
    known_community_cards=['QH', 'JH', '10H'],
    already_dropped_cards=['2S', '4D', 'AC', '5C']
)

# [0.2558139534883721, 0.636766334440753, 0.10741971207087486]
probabilities = count_win_probabilities(
    players_private_cards=[('2C', 'QC'), ('KH', 'KS'), ('AD', '8D')],
    known_community_cards=['3C', '6H', 'QS'],
    already_dropped_cards=[]
)
```

## Other
Check out the [tests generator](https://github.com/eeeeeeeelias/holdem-win-probability/blob/trunk/src/holdem_probability/generator_of_tests.py).

## License
[MIT](https://choosealicense.com/licenses/mit/)
