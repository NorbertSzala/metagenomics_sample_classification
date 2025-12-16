import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_roc_curves(predictions_file, ground_truth_file, output_file=None):
    """
    Plot ROC curves and calculate mean AUC.
    
    Args:
        predictions_file: Path to predictions TSV
        ground_truth_file: Path to ground truth TSV
        output_file: Path to save plot (if None, don't save)
        
    Returns:
        mean_auc: Mean AUC-ROC score (or None if error)
    """
    
    # ---- Load predictions ----
    predictions = {}

    with open(predictions_file) as f:
        reader = csv.DictReader(f, delimiter='\t')

        if "fasta_file" not in reader.fieldnames:
            print("ERROR: 'fasta_file' column missing in predictions TSV")
            return None

        classes = [c for c in reader.fieldnames if c != "fasta_file"]

        for row in reader:
            sample = Path(row["fasta_file"]).name
            try:
                predictions[sample] = {cls: float(row[cls]) for cls in classes}
            except ValueError:
                print(f"WARNING: non-numeric score in row: {row}")
                continue

    # ---- Load ground truth ----
    ground_truth = {}

    with open(ground_truth_file) as f:
        reader = csv.DictReader(f, delimiter='\t')

        if "fasta_file" not in reader.fieldnames or "geo_loc_name" not in reader.fieldnames:
            print("ERROR: ground truth TSV must contain 'fasta_file' and 'geo_loc_name'")
            return None

        for row in reader:
            sample = Path(row["fasta_file"]).name
            ground_truth[sample] = row["geo_loc_name"]

    # ---- Check overlap ----
    common = set(predictions) & set(ground_truth)

    if len(common) == 0:
        print("ERROR: No overlapping sample names between predictions and ground truth")

    # ---- Calculate ROC for each class ----
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
            continue

        fpr, tpr, auc = calculate_roc_curve(y_true, y_scores)
        all_aucs.append(auc)
        
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc:.3f})", linewidth=2)

    mean_auc = np.mean(all_aucs)
    # ---- Final plot formatting ----
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.5)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curves – Mean AUC: {mean_auc:.3f}", fontsize=14, fontweight='bold')
    plt.legend(fontsize=8, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()

    # ---- Save plot (if requested) ----
    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.close()

    return mean_auc


def calculate_roc_curve(y_true, y_scores):
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot ROC curves and calculate AUC')
    parser.add_argument('predictions', help='Predictions TSV file')
    parser.add_argument('ground_truth', help='Ground truth TSV file')
    parser.add_argument('-o', '--output', help='Output plot file (PNG)', default=None)
    
    args = parser.parse_args()
    
    mean_auc = plot_roc_curves(args.predictions, args.ground_truth, args.output)