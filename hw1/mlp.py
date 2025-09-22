# helpers_vector_linear.py
from reverse_mode import variable, exp, gradient  # ops are overloaded on cobundle

def dot(u, v):
    # u, v: lists of cobundles (same length)
    s = variable(0.0)
    for ui, vi in zip(u, v):
        s = s + ui * vi
    return s  # cobundle

def matvec(W, x):
    # W: list[list[cobundle]], x: list[cobundle]
    return [dot(row, x) for row in W]  # list[cobundle]

def vadd(u, v):
    # elementwise add two vectors (lists of cobundles)
    return [ui + vi for ui, vi in zip(u, v)]

def linear(x, W, b):
    # Wx + b, all cobundles
    return vadd(matvec(W, x), b)


# toy params and input
W = [
    [variable(1.0), variable(2.0)],
    [variable(3.0), variable(4.0)]
]
b = [variable(0.5), variable(-1.0)]
x = [variable(0.1), variable(0.2)]

y = linear(x, W, b)
print([yi.prim for yi in y])   # expect [1.0, 0.1]


def sigmoid_scalar(z):
    # z is a cobundle
    return 1 / (1 + exp(-z))

def sigmoid_vec(z_vec):
    # list[cobundle] -> list[cobundle]
    return [sigmoid_scalar(zi) for zi in z_vec]

def slp(x, W, b):
    # Single layer perceptron: sigmoid(Wx + b)
    return sigmoid_vec(linear(x, W, b))


W = [
    [variable(1.0), variable(2.0)],
    [variable(3.0), variable(4.0)]
]
b = [variable(0.5), variable(-1.0)]
x = [variable(0.1), variable(0.2)]

h = slp(x, W, b)
print([hi.prim for hi in h])
# Expected (approximately): [0.7310585786, 0.5249791875]
# because linear prims were [1.0, 0.1], and sigmoid(1.0)≈0.73106, sigmoid(0.1)≈0.52498


def l2_loss(y_hat_vec, y_vec):
    # sum_i (y_hat[i] - y[i])^2  -> returns a scalar cobundle
    s = variable(0.0)
    for yh, y in zip(y_hat_vec, y_vec):
        d = yh - y
        s = s + d*d
    return s

def mlp(x, params):
    """
    params: list of (W,b) pairs
      W is list[list[cobundle-like]]
      b is list[cobundle-like]
    """
    h = x
    for (W, b) in params:
        h = slp(h, W, b)
    return h  # vector (list of cobundles)


# packing.py
def build_params_from_theta(theta, layer_shapes):
    """
    layer_shapes: list of tuples (out_dim, in_dim) for each layer
    returns: params = [(W1,b1), (W2,b2), ...]
    W: out_dim x in_dim  (list of rows)
    b: length out_dim
    """
    i = 0
    params = []
    for (out_dim, in_dim) in layer_shapes:
        W = []
        for _ in range(out_dim):
            row = []
            for _ in range(in_dim):
                row.append(theta[i]); i += 1
            W.append(row)
        b = []
        for _ in range(out_dim):
            b.append(theta[i]); i += 1
        params.append((W, b))
    assert i == len(theta), "Theta length mismatch with layer_shapes."
    return params


# 1) Model shape: one layer, out=1, in=2
layer_shapes = [(1, 2)]
# theta has size: out*in + out = 1*2 + 1 = 3
theta0 = [0.5, -0.3, 0.1]  # [W00, W01, b0] in row-major for the single row

# 2) One training sample (e.g., XOR case: x=[0,1] -> y=[1])
x_sample = [0.0, 1.0]
y_sample = [1.0]

def sample_loss(theta):
    # theta is either floats (forward) or cobundles (during gradient wrap)
    params = build_params_from_theta(theta, layer_shapes)
    y_hat = mlp(x_sample, params)   # vector length 1
    return l2_loss(y_hat, y_sample) # scalar cobundle (required for reverse-mode)

# 3) Reverse-mode gradient wrt theta
g = gradient(sample_loss)(theta0)   # returns list of ∂loss/∂theta_i

print("theta0:", theta0)
print("grad :", g)

# 4) One tiny descent step to sanity-check sign
eta = 1e-2
theta1 = [t - eta*gi for t, gi in zip(theta0, g)]

# Recompute loss before/after
L0 = sample_loss(theta0).prim
L1 = sample_loss(theta1).prim
print("Loss before:", L0)
print("Loss after :", L1)


def dataset_loss(theta, layer_shapes, dataset):
    params = build_params_from_theta(theta, layer_shapes)
    total = 0.0
    for x, y in dataset:
        y_hat = mlp(x, params)       # vector of cobundles
        total += l2_loss(y_hat, y).prim  # scalar -> take prim to sum as float
    return total / len(dataset)

