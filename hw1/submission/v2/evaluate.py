# evaluate.py
from utils import mlp, l2_loss, build_params_from_theta

def average_loss(params, dataset):
    total = 0.0
    for x, y in dataset:
        y_hat = mlp(x, params)
        total += l2_loss(y_hat, y).prim
    return total / max(1, len(dataset))

def accuracy01(params, dataset, threshold=0.5):
    from mlp import mlp
    correct = 0
    for x, y in dataset:
        y_hat = [t.prim for t in mlp(x, params)]
        y_bin = [1.0 if v >= threshold else 0.0 for v in y_hat]
        if y_bin == y:
            correct += 1
    return correct / max(1, len(dataset))
