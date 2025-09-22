# train_mlp_adder_demo.py
import random, time
from datasets import two_bit_adder_dataset
from splits import train_test_split
from mlp import build_params_from_theta
from evaluate import average_loss, accuracy01
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
    last_it=0; last_L=None; last_theta=theta0
    for it, L, th in train_gd(theta0, layer_shapes, train, lr, iters, log_every=max(1,iters//5 or 1)):
        last_it, last_L, last_theta = it, L, th
    secs = time.time() - start
    params = build_params_from_theta(last_theta, layer_shapes)
    test_L = average_loss(params, test)
    # test_acc = accuracy01(params, test)
    return {"iters": last_it, "train_L": last_L, "test_L": test_L, 
            # "acc": test_acc, 
            "secs": secs}

def main():
    data = two_bit_adder_dataset()
    grids = [
        {"layer_shapes":[(2,3),(4,2)], "lr":0.5, "iters":800},
        {"layer_shapes":[(4,3),(4,4)], "lr":0.7, "iters":1200},
        {"layer_shapes":[(5,5),(5,5)], "lr":0.3, "iters":2000},
    ]
    splits = [(0.5,0), (0.75,1), (0.8,2)]
    print("dataset=TwoBitAdder")
    for ls in grids:
        print(f"\nmodel={ls['layer_shapes']}  lr={ls['lr']}  iters={ls['iters']}")
        print("split\ttrain_L\ttest_L"
              # "\tacc"
              "\tsecs")
        for frac, seed in splits:
            train, test = train_test_split(data, train_frac=frac, seed=seed)
            r = run_one(train, test, ls["layer_shapes"], ls["lr"], ls["iters"], seed=seed)
            print(f"{int(frac*100)}-{seed}\t{r['train_L']:.4f}\t{r['test_L']:.4f}"
                  # f"\t{r['acc']:.2f}"
                  f"\t{r['secs']:.2f}")

if __name__ == "__main__":
    main()
