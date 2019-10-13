import random


def flip_coin(p):
    """
    Return true with probability p

    Args:
        p: float, probability to return True

    Returns: bool

    """

    assert 0 <= p <= 1, 'A probability should be between 0 and 1'
    return random.random() < p
