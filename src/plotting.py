def plot_roc_curves(predictions_file, ground_truth_file, output_file=None):
    import csv
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import argparse
    import numpy as np


    # ---- Load predictions ----
    predictions = {}

    with open(predictions_file) as f:
        reader = csv.DictReader(f, delimiter='\t')

        if "fasta_file" not in reader.fieldnames:
            print("ERROR: 'fasta_file' column missing in predictions TSV")
            return

        classes = [c for c in reader.fieldnames if c != "fasta_file"]

        for row in reader:
            sample = Path(row["fasta_file"]).name
            try:
                predictions[sample] = {cls: float(row[cls]) for cls in classes}
            except ValueError:
                print(f"WARNING: non-numeric score in row: {row}")
    

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



    # ---- Check overlap ----
    common = set(predictions) & set(ground_truth)

    if len(common) == 0:
        print("ERROR: No overlapping sample names between predictions and ground truth")

    # ---- Plot ----
    plt.figure(figsize=(12, 10))
    all_aucs = []


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

        if n_pos == 0 or n_neg == 0:
            print("  WARNING: ROC undefined (no positives or no negatives)")
            continue

        fpr, tpr, auc = calculate_roc_curve(y_true, y_scores)
        all_aucs.append(auc)

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

    if output_file is not None:
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
    else:
        plt.close()


    if output_file is not None:
        print(f"\nROC curves saved to: {output_file}")

    
    
    mean_auc = float(np.mean(all_aucs))

    if output_file is not None:
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    return mean_auc




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