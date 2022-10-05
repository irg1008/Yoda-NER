import random as rnd
import numpy as np


def batch_generator(X, Y, batch_size=4, pad=None, shuffle=True):
    """Generate batches of data.
    Args:
        X (list): List of sentences.
        Y (list): List of labels.
        pad_value: Value use for padding. Will be removed on trainning. Necessary for regular shaped batches.
        batch_size (int, optional): Size of the batch. Defaults to 4.
        shuffle (bool, optional): Shuffle data. Defaults to True.
    Yields:
        tuple: Batch of data.
    """
    if shuffle:
        data = list(zip(X, Y))
        rnd.shuffle(data)
        X, Y = zip(*data)

    max_input_size = max(len(x) for x in X)

    def set(batch_idx, global_idx, batch, data):
        item = data[global_idx]
        batch[i, : len(item)] = item

    data_idx = 0
    while data_idx < len(X):
        x = np.full((batch_size, max_input_size), fill_value=pad)
        y = x.copy()

        for i in range(batch_size):
            set(i, data_idx, x, X)
            set(i, data_idx, y, Y)

            data_idx += 1
            if data_idx >= len(X):
                break

        yield x, y
