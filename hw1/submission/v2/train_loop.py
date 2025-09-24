# train_loop.py
from reverse_mode import gradient
from utils import build_params_from_theta, mlp, l2_loss
from evaluate import average_loss

def make_objective(layer_shapes, train):
    def f(theta):
        params = build_params_from_theta(theta, layer_shapes)
        s = 0.0
        for x, y in train:
            s = s + l2_loss(mlp(x, params), y)
        return s * (1.0 / max(1, len(train)))
    return f

def train_gd(theta, layer_shapes, train, lr, iters, log_every=100):
    f = make_objective(layer_shapes, train)
    g = gradient(f)
    for t in range(iters):
        theta = [p - lr*gi for p, gi in zip(theta, g(theta))]
        if (t % log_every) == 0:
            yield t, f(theta).prim, theta
    yield iters, f(theta).prim, theta  # final
