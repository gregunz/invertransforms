import random


def flip_coin(p):
    assert 0 <= p <= 1, 'A probability should be between 0 and 1'
    return random.random() < p
