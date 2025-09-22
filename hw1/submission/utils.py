from reverse_mode import variable, exp, gradient

def dot(u, v):
    s = variable(0.0)
    for ui, vi in zip(u, v):
        s = s + ui * vi
    return s

def matvec(W, x):
    return [dot(row, x) for row in W]

def vadd(u, v):
    return [ui + vi for ui, vi in zip(u, v)]

def linear(x, W, b):
    return vadd(matvec(W, x), b)

def sigmoid_scalar(z):
    return 1 / (1 + exp(-z))

def sigmoid_vec(z_vec):
    return [sigmoid_scalar(zi) for zi in z_vec]

def slp(x, W, b):
    return sigmoid_vec(linear(x, W, b))

def l2_loss(y_hat_vec, y_vec):
    s = variable(0.0)
    for yh, y in zip(y_hat_vec, y_vec):
        d = yh - y
        s = s + d*d
    return s

def mlp(x, params):
    h = x
    for (W, b) in params:
        h = slp(h, W, b)
    return h

def build_params_from_theta(theta, layer_shapes):
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

def dataset_loss(theta, layer_shapes, dataset):
    params = build_params_from_theta(theta, layer_shapes)
    total = 0.0
    for x, y in dataset:
        y_hat = mlp(x, params)
        total += l2_loss(y_hat, y).prim
    return total / len(dataset)
