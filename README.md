# holdem-win-probability

![Python >= 3.6](https://img.shields.io/badge/python->=3.6-blue) ![Python >= 3.6](https://img.shields.io/badge/license-MIT-green)

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

## License
[MIT](https://choosealicense.com/licenses/mit/)
