# train_mlp_xor_demo.py
import time, random
from datasets import xor_dataset
from splits import train_test_split
from evaluate import average_loss
from utils import build_params_from_theta
from train_loop import train_gd

def make_theta(layer_shapes, seed=0, scale=0.5):
    rng = random.Random(seed)
    theta=[]
    for (out_dim, in_dim) in layer_shapes:
        for _ in range(out_dim*in_dim): theta.append((rng.random()*2-1)*scale)
        for _ in range(out_dim):        theta.append((rng.random()*2-1)*scale)
    return theta

def run_one(train, test, layer_shapes, lr, iters, seed=0):
    theta0 = make_theta(layer_shapes, seed=seed)
    start = time.time()
    last_iter = 0; last_loss = None; last_theta = theta0
    print(f"iter; loss")
    for it, L, th in train_gd(theta0, layer_shapes, train, lr, iters, log_every=max(1,iters//5 or 1)):
        last_iter, last_loss, last_theta = it, L, th
        print(f"{last_iter}; {last_loss}")
    dur = time.time() - start
    params = build_params_from_theta(last_theta, layer_shapes)
    test_L = average_loss(params, test)
    return {"iters": last_iter, "train_L": last_loss, "test_L": test_L, "secs": dur}

def main():
    data = xor_dataset()
    grids = [
        {"layer_shapes":[(2,2),(1,2)], "lr":0.5, "iters":400},
        {"layer_shapes":[(3,2),(1,3)], "lr":1.0, "iters":800},
        {"layer_shapes":[(4,2),(3,4),(1,3)], "lr":0.3, "iters":1200},
    ]
    splits = [(0.5,0), (0.75,1), (0.9,2)]
    print("dataset=XOR")
    for ls in grids:
        print(f"\n\nmodel={ls['layer_shapes']}  lr={ls['lr']}  iters={ls['iters']}")
        for frac, seed in splits:
            train, test = train_test_split(data, train_frac=frac, seed=seed)
            r = run_one(train, test, ls["layer_shapes"], ls["lr"], ls["iters"], seed=seed)
            print("split-seed\ttrain_L\ttest_L\tsecs")
            print(f"{int(frac*100)}-{seed}\t{r['train_L']:.4f}\t{r['test_L']:.4f}\t{r['secs']:.2f}")
            print("\n")

if __name__ == "__main__":
    main()
