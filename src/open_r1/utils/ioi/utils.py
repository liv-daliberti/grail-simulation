"""Miscellaneous helpers shared by the IOI scoring utilities."""

from itertools import islice


def batched(iterable, batch_size):
    "Batch data into lists of length ``batch_size``. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if batch_size < 1:
        yield list(iterable)
        return

    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch
