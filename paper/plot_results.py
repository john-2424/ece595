import os
import csv
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)


def load_log(path):
    epochs, train_acc, val_acc = [], [], []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # epoch, train_loss, train_acc, val_loss, val_acc
        for row in reader:
            epoch = int(row[0])
            tr_acc = float(row[2])
            v_acc = float(row[4])
            epochs.append(epoch)
            train_acc.append(tr_acc)
            val_acc.append(v_acc)
    return epochs, train_acc, val_acc


def plot_two_experiments(exp1_name, exp1_label, exp2_name, exp2_label, title, outfile):
    path1 = os.path.join(RESULTS_DIR, f"{exp1_name}_log.csv")
    path2 = os.path.join(RESULTS_DIR, f"{exp2_name}_log.csv")

    e1, _, v1 = load_log(path1)
    e2, _, v2 = load_log(path2)

    plt.figure()
    plt.plot(e1, v1, label=exp1_label)
    plt.plot(e2, v2, label=exp2_label)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, outfile))
    plt.close()


def plot_multi_experiment(exp_names, labels, title, outfile):
    plt.figure()
    for exp_name, label in zip(exp_names, labels):
        path = os.path.join(RESULTS_DIR, f"{exp_name}_log.csv")
        epochs, _, val_acc = load_log(path)
        plt.plot(epochs, val_acc, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, outfile))
    plt.close()


def main():
    # Names must match exp_name in train_cifar10.py

    # Baseline comparison: ResNet-18 vs ConvNeXt-Tiny
    resnet_exp = "resnet18_stempatchify_k7_ln_mlp4_cifar10"
    convnext_base_exp = "convnext_tiny_stempatchify_k7_ln_mlp4_cifar10"
    plot_two_experiments(
        resnet_exp,
        "ResNet-18",
        convnext_base_exp,
        "ConvNeXt-Tiny",
        "Baseline: ResNet-18 vs ConvNeXt-Tiny on CIFAR-10",
        "baseline_resnet_vs_convnext.png",
    )

    # Kernel ablation: k3 vs k7
    k3_exp = "convnext_tiny_stempatchify_k3_ln_mlp4_cifar10"
    k7_exp = "convnext_tiny_stempatchify_k7_ln_mlp4_cifar10"
    plot_two_experiments(
        k3_exp,
        "ConvNeXt-Tiny k=3",
        k7_exp,
        "ConvNeXt-Tiny k=7",
        "Kernel Size Ablation (k=3 vs k=7)",
        "ablation_kernel_k3_vs_k7.png",
    )

    # Norm ablation: LN vs BN
    ln_exp = "convnext_tiny_stempatchify_k7_ln_mlp4_cifar10"
    bn_exp = "convnext_tiny_stempatchify_k7_bn_mlp4_cifar10"
    plot_two_experiments(
        ln_exp,
        "LN",
        bn_exp,
        "BN",
        "Normalization Ablation (LN vs BN)",
        "ablation_norm_ln_vs_bn.png",
    )

    # MLP ratio ablation: mlp2, mlp4, mlp6
    mlp2_exp = "convnext_tiny_stempatchify_k7_ln_mlp2_cifar10"
    mlp4_exp = "convnext_tiny_stempatchify_k7_ln_mlp4_cifar10"
    mlp6_exp = "convnext_tiny_stempatchify_k7_ln_mlp6_cifar10"
    plot_multi_experiment(
        [mlp2_exp, mlp4_exp, mlp6_exp],
        ["mlp_ratio=2", "mlp_ratio=4", "mlp_ratio=6"],
        "MLP Ratio Ablation",
        "ablation_mlp_ratio.png",
    )

    # Stem ablation: patchify vs resnet
    stem_patchify_exp = "convnext_tiny_stempatchify_k7_ln_mlp4_cifar10"
    stem_resnet_exp = "convnext_tiny_stemresnet_k7_ln_mlp4_cifar10"
    plot_two_experiments(
        stem_patchify_exp,
        "Patchify Stem",
        stem_resnet_exp,
        "ResNet-style Stem",
        "Stem Type Ablation (Patchify vs ResNet-style)",
        "ablation_stem_patchify_vs_resnet.png",
    )

    print(f"Plots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
