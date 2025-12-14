import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_roc_curves(predictions_file, ground_truth_file, output_file='roc_curves.png'):
    """
    Plot ROC curves for all classes.
    """
    import csv
    
    # Load data
    predictions = {}
    classes = []
    
    with open(predictions_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        classes = [col for col in reader.fieldnames if col != 'fasta_file']
        
        for row in reader:
            sample = row['fasta_file']
            predictions[sample] = {cls: float(row[cls]) for cls in classes}
    
    ground_truth = {}
    with open(ground_truth_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            ground_truth[row['fasta_file']] = row['geo_loc_name']
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    all_aucs = []
    
    # Plot ROC curve for each class
    for class_name in sorted(classes):
        # Create binary labels
        y_true = []
        y_scores = []
        
        for sample in predictions.keys():
            if sample in ground_truth:
                y_true.append(1 if ground_truth[sample] == class_name else 0)
                y_scores.append(predictions[sample][class_name])
        
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # Calculate ROC curve
        fpr, tpr, auc = calculate_roc_curve(y_true, y_scores)
        all_aucs.append(auc)
        
        # Plot
        plt.plot(fpr, tpr, label=f'{class_name} (AUC={auc:.3f})', linewidth=2)
    
    # Add diagonal reference line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5)')
    
    # Formatting
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - Mean AUC: {np.mean(all_aucs):.3f}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {output_file}")
    
    plt.show()


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
