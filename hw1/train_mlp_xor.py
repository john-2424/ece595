# train_xor.py
import random
from reverse_mode import gradient
from datasets import xor_dataset
from mlp import build_params_from_theta
from mlp import mlp, l2_loss

def make_theta_random(layer_shapes, seed=0, scale=0.5):
    random.seed(seed)
    theta = []
    for (out_dim, in_dim) in layer_shapes:
        # W entries, then b entries
        for _ in range(out_dim * in_dim):
            theta.append((random.random()*2-1)*scale)
        for _ in range(out_dim):
            theta.append((random.random()*2-1)*scale)
    return theta

def xor_loss_fn(layer_shapes, dataset):
    # returns a callable f(theta)->scalar cobundle for reverse-mode
    def f(theta):
        params = build_params_from_theta(theta, layer_shapes)
        s = 0.0
        for x, y in dataset:
            y_hat = mlp(x, params)         # vector of cobundles
            s = s + l2_loss(y_hat, y)      # keep as cobundle-summed
        # average
        m = len(dataset)
        return s * (1.0 / m)
    return f

def train_xor():
    # network: 2 -> 2 -> 1  (good for XOR)
    layer_shapes = [(2, 2), (1, 2)]
    data = xor_dataset()

    theta = make_theta_random(layer_shapes, seed=42, scale=0.5)
    f = xor_loss_fn(layer_shapes, data)
    grad_f = gradient(f)

    eta = 1.0          # try 1.0, then adjust if needed
    iters = 10000       # keep small; you can bump later

    print("iter\tloss")
    for t in range(iters+1):
        if t % 100 == 0:
            # read loss value (prim) at current theta
            L = f(theta).prim
            print(f"{t}\t{L:.6f}")
        g = grad_f(theta)               # list of dL/dtheta_i
        theta = [p - eta*gi for p, gi in zip(theta, g)]

    # quick sanity: print predictions
    params = build_params_from_theta(theta, layer_shapes)
    for x, y in data:
        y_hat = mlp(x, params)[0].prim
        print(f"x={x}  y={y[0]}  y_hat={y_hat:.4f}")

if __name__ == "__main__":
    train_xor()
