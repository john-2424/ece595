# splits.py
import random

def train_test_split(dataset, train_frac=0.5, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(dataset)))
    rng.shuffle(idx)
    k = int(round(train_frac * len(dataset)))
    train = [dataset[i] for i in idx[:k]]
    test  = [dataset[i] for i in idx[k:]]
    return train, test

def kfold_splits(dataset, k=4, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(dataset)))
    rng.shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    for i in range(k):
        test_idx = set(folds[i])
        train = [dataset[j] for j in idx if j not in test_idx]
        test  = [dataset[j] for j in folds[i]]
        yield train, test
