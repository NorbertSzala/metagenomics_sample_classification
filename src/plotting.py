def plot_roc_curves(predictions_file, ground_truth_file, output_file='roc_curves.png'):
    import csv
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import argparse
    import numpy as np
    print("=== DEBUG: LOADING DATA ===")
    print(f"Predictions file: {predictions_file}")
    print(f"Ground truth file: {ground_truth_file}")

    # ---- Load predictions ----
    predictions = {}

    with open(predictions_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        print("Prediction columns:", reader.fieldnames)

        if "fasta_file" not in reader.fieldnames:
            print("ERROR: 'fasta_file' column missing in predictions TSV")
            return

        classes = [c for c in reader.fieldnames if c != "fasta_file"]
        print(f"Detected classes ({len(classes)}):", classes)

        for row in reader:
            sample = Path(row["fasta_file"]).name
            try:
                predictions[sample] = {cls: float(row[cls]) for cls in classes}
            except ValueError:
                print(f"WARNING: non-numeric score in row: {row}")
    
    print(f"Loaded {len(predictions)} prediction samples")

    # ---- Load ground truth ----
    ground_truth = {}

    with open(ground_truth_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        print("Ground truth columns:", reader.fieldnames)

        if "fasta_file" not in reader.fieldnames or "geo_loc_name" not in reader.fieldnames:
            print("ERROR: ground truth TSV must contain 'fasta_file' and 'geo_loc_name'")
            return

        for row in reader:
            sample = Path(row["fasta_file"]).name
            ground_truth[sample] = row["geo_loc_name"]


    print(f"Loaded {len(ground_truth)} ground truth samples")

    # ---- Check overlap ----
    common = set(predictions) & set(ground_truth)
    print(f"Samples in common: {len(common)}")

    if len(common) == 0:
        print("ERROR: No overlapping sample names between predictions and ground truth!")
        print("Example prediction keys:", list(predictions.keys())[:3])
        print("Example ground truth keys:", list(ground_truth.keys())[:3])
        return

    # ---- Plot ----
    plt.figure(figsize=(12, 10))
    all_aucs = []

    print("\n=== DEBUG: PER-CLASS ANALYSIS ===")

    for class_name in sorted(classes):
        y_true = []
        y_scores = []

        for sample in common:
            y_true.append(1 if ground_truth[sample] == class_name else 0)
            y_scores.append(predictions[sample][class_name])

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        print(f"\nClass: {class_name}")
        print(f"  positives: {n_pos}, negatives: {n_neg}")
        print(f"  score range: min={y_scores.min():.4f}, max={y_scores.max():.4f}")
        print(f"  score mean: {y_scores.mean():.4f}")

        if n_pos == 0 or n_neg == 0:
            print("  WARNING: ROC undefined (no positives or no negatives)")
            continue

        fpr, tpr, auc = calculate_roc_curve(y_true, y_scores)
        all_aucs.append(auc)

        print(f"  AUC: {auc:.4f}")

        plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc:.3f})")

    if len(all_aucs) == 0:
        print("ERROR: No valid ROC curves could be computed")
        return

    print(f"\nMean AUC: {np.mean(all_aucs):.4f}")

    # ---- Final plot formatting ----
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves – Mean AUC: {np.mean(all_aucs):.3f}")
    plt.legend(fontsize=8)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"\nROC curves saved to: {output_file}")




def calculate_roc_curve(y_true, y_scores):
    import numpy as np
    """Calculate FPR, TPR, and AUC."""
    
    thresholds = np.unique(y_scores)
    thresholds = np.sort(thresholds)[::-1]
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.5
    
    tpr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
    
    fpr = np.array([0.0] + fpr_list + [1.0])
    tpr = np.array([0.0] + tpr_list + [1.0])
    
    # Calculate AUC
    sorted_idx = np.argsort(fpr)
    fpr_sorted = fpr[sorted_idx]
    tpr_sorted = tpr[sorted_idx]
    
    auc = 0.0
    for i in range(len(fpr_sorted) - 1):
        width = fpr_sorted[i + 1] - fpr_sorted[i]
        height = (tpr_sorted[i] + tpr_sorted[i + 1]) / 2
        auc += width * height
    
    return fpr, tpr, auc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_tsv")
    parser.add_argument("ground_truth_tsv")
    parser.add_argument('output_name', type=int, default = 'roc_curves.png')

    args = parser.parse_args()
    
    plot_roc_curves(
        args.prediction_tsv,
        args.ground_truth_tsv,
        args.output_name
    )
