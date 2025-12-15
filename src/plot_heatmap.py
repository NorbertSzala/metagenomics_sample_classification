import numpy as np
import matplotlib.pyplot as plt
import csv



def plot_auc_heatmap(results_tsv: str, output_png: str = "results/auc_heatmap.png"):
    Ks = set()
    Ss = set()
    data = []

    # ---- load results ----
    with open(results_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            k = int(row["K"])
            s = int(row["sketch_size"])
            auc = float(row["mean_auc"])

            Ks.add(k)
            Ss.add(s)
            data.append((k, s, auc))

    Ks = sorted(Ks)
    Ss = sorted(Ss)

    auc_matrix = np.full((len(Ks), len(Ss)), np.nan)

    for k, s, auc in data:
        i = Ks.index(k)
        j = Ss.index(s)
        auc_matrix[i, j] = auc

    # ---- find best parameters ----
    best_idx = np.nanargmax(auc_matrix)
    best_i, best_j = np.unravel_index(best_idx, auc_matrix.shape)

    best_k = Ks[best_i]
    best_s = Ss[best_j]
    best_auc = auc_matrix[best_i, best_j]

    # ---- plot ----
    plt.figure(figsize=(12, 8))
    im = plt.imshow(auc_matrix, aspect="auto")

    plt.colorbar(im, label="Mean AUC")
    plt.xticks(range(len(Ss)), Ss, rotation=45)
    plt.yticks(range(len(Ks)), Ks)

    plt.xlabel("Sketch size")
    plt.ylabel("K")
    plt.title("AUC heatmap (K × sketch_size)")

    # ---- mark best point ----
    plt.scatter(
        best_j, best_i,
        s=200,
        facecolors="none",
        edgecolors="red",
        linewidths=2,
        label="Best parameters"
    )

    plt.text(
        best_j, best_i,
        f"AUC={best_auc:.3f}",
        color="red",
        ha="center",
        va="bottom",
        fontsize=10,
        weight="bold"
    )

    plt.legend()

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"Heatmap saved to {output_png}")
    print(
        f"Best parameters from heatmap: "
        f"K={best_k}, sketch_size={best_s}, mean AUC={best_auc:.4f}"
    )

    return best_k, best_s, best_auc


best_k, best_s, best_auc = plot_auc_heatmap(
    results_tsv="results/optimization_results.tsv"
)

print(f"Chosen parameters: K={best_k}, sketch_size={best_s}, AUC={best_auc:.3f}")