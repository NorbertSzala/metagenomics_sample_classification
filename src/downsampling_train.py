import random
import csv, sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from main import main
from plotting import plot_roc_curves


def downsample_tsv(
    
    tsv_path: str,
    fraction: float,
    output_tsv: str
):
    
    """
    Create a downsampled version of a test TSV file.

    Function randomly selects a given fraction of test samples
    and writes them to a new TSV file. All FASTA paths
    are converted to absolute paths to avoid path errors
    when the TSV is moved to a different directory.

    Args:
        test_tsv (str): Path to the original test TSV file.
        fraction (float): Fraction of samples to keep (e.g. 0.2 = 20%).
        output_tsv (str): Path where thenew downsampled TSV will be written.
    """

    tsv_path = Path(tsv_path).resolve()
    base_dir = tsv_path.parent  

    with open(tsv_path) as f:
        reader = list(csv.DictReader(f, delimiter="\t"))

    # number of samples to keep after downsampling
    n_total = len(reader)
    n_keep = max(1, int(fraction * n_total))
    
    # randomly selected samples without replacement
    sampled = random.sample(reader, n_keep)

    # absolute fasta path
    for row in sampled:
        fasta_path = Path(row["fasta_file"])
        if not fasta_path.is_absolute():
            row["fasta_file"] = str((base_dir / fasta_path).resolve())

    # Save output
    with open(output_tsv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=reader[0].keys(),
            delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(sampled)



def evaluate_downsampled(
    
    training_tsv: str,
    testing_tsv: str,
    ground_truth_tsv: str,
    data_to_downsize: str,
    fractions=np.arange(0.0, 1.1, 0.1),
    n_repeats: int = 5,
    output_dir: str = "results/downsampling_test"
):
    """
    For each test fraction, the test set is randomly downsampled
    multiple times. The model is evaluated on each subset, and
    the mean AUC and standard deviation are computed.

    This procedure estimates the variance of model performance taking into account
    the choice of test samples.

    Args:
        training_tsv (str): Path to the training TSV file.
        test_tsv (str): Path to the full test TSV file.
        ground_truth_tsv (str): Path to the ground truth labels TSV.
        fractions (tuple): Fractions of test data to evaluate.
        n_repeats (int): Number of random repetitions per fraction.
        output_dir (str): Directory for temporary TSVs and predictions.

    Returns:
        dict: Mapping fraction -> {"mean": mean_auc, "std": std_auc}.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    total_steps = len(fractions) * n_repeats

    # nice looking progress bar
    with tqdm(total=total_steps, desc="Downsampling test set", unit="run") as pbar:

        for frac in fractions:
            aucs = []

            for rep in range(n_repeats):
                # temporary files
                tmp_path = output_dir / f"{data_to_downsize}_{int(frac*100)}pct_rep{rep}.tsv"
                pred_tsv = output_dir / f"pred_{int(frac*100)}pct_rep{rep}.tsv"
                

                # downsampled test TSV
                # and run the full classification pipeline - on the same training data
                if data_to_downsize == "test":
                    downsample_tsv(testing_tsv, frac, tmp_path)
                    main(training_tsv, str(tmp_path), str(pred_tsv), ground_truth_tsv=None)

                else: 
                    downsample_tsv(training_tsv, frac, tmp_path)
                    main(str(tmp_path), testing_tsv, str(pred_tsv), ground_truth_tsv=None)
  
                # compute mean AUC. doesnt make plot
                mean_auc = plot_roc_curves(
                    predictions_file=str(pred_tsv),
                    ground_truth_file=ground_truth_tsv,
                    output_file=None
                )

                aucs.append(mean_auc)

                # ---- update progress bar ----
                pbar.set_postfix({
                    "test_%": int(frac * 100),
                    "rep": rep + 1,
                    "AUC": f"{mean_auc:.3f}"
                })
                pbar.update(1)

            summary[frac] = {
                "mean": float(np.mean(aucs)),
                "std": float(np.std(aucs))
            }

    return summary


def plot_downsampling_results(summary, data_to_downsize, output_png):
    """
    Plot the effect of test set downsampling on model performance.

    The plot shows mean AUC with standard deviation error bars on Y axis 
    and fraction  of test data used on X axis.

    Args:
        summary (dict): Output from evaluate_downsampled_test().
        output_png (str): Path to the output PNG file.
    """
  
    fracs = sorted(summary.keys())
    means = [summary[f]["mean"] for f in fracs]
    stds = [summary[f]["std"] for f in fracs]

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        [f * 100 for f in fracs],
        means,
        yerr=stds,
        fmt="o-",
        capsize=5
    )

    plt.xlabel(f"{data_to_downsize} data used (%)")
    plt.ylabel("Mean AUC")
    plt.title(f"Downsampling {data_to_downsize} set: mean AUC ± SD")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"Downsampling plot saved to {output_png}")


if __name__ == "__main__":

    options = ['train', 'test']

    if len(sys.argv) != 2 or sys.argv[1].lower() not in options:
        print("Usage: python downsampling.py data_to_downsize(print: train or test)", file=sys.stderr)
        sys.exit(1)

    option = sys.argv[1].lower()

    summary = evaluate_downsampled(
        training_tsv = "data/train0_data.tsv",
        test_tsv = "data/test0_data.tsv",
        ground_truth_tsv = "data/test0_ground_truth.tsv",
        data_to_downsize = option,
        fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        n_repeats = 5
    )
    output_png = f"results/downsampling_{option}_auc.png"
    plot_downsampling_results(summary, option, output_png)