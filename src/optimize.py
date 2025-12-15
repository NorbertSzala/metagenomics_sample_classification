# Norbert - find the best parameters to model

from main import main
import numpy as np
from pathlib import Path
import csv
from plotting import plot_roc_curves
from logs import RunLogger
from tqdm import tqdm

def optimize_parameters(
    training_tsv: str,
    testing_tsv: str,
    ground_truth_tsv: str,
    param_grid: dict,
    output_results_tsv: str = "results/optimization_results.tsv"
):
    """
    Scaffold of function finding the most optimal parameters, according to accuracy, time usage and memory usage in main function

    Args:
        training_tsv (str): path to training set
        testing_tsv (str): path to testing set
        param_grid (dict): dictionary with list of K and SKETCH_SIZE values

    Returns:
        list: list of parameters from input with statistics 
    """
    results = []
    best_parameters = None
    best_auc = -1.0   # initialize with impossible low AUC

    output_results_tsv = Path(output_results_tsv)
    output_results_tsv.parent.mkdir(exist_ok=True)
    total_runs = len(param_grid["K"]) * len(param_grid["sketch_size"])

        # ---- open file ONCE ----
    with open(output_results_tsv, "w", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t")

        writer.writerow([
            "K",
            "sketch_size",
            "mean_auc",
            "total_time_sec",
            "peak_ram_mb"
        ])


        # ---- progress bar ----
        with tqdm(total=total_runs, desc="Grid search", unit="run") as pbar:
            for k in param_grid["K"]:
                for s in param_grid["sketch_size"]:

                    print(f"\n=== RUN: K={k}, sketch_size={s} ===")

                    pred_tsv = Path("results") / f"pred_K{k}_S{s}.tsv"

                    # ---- RUN MODEL (this CREATES pred_tsv and returns stats) ----
                    run_stats = main(
                        training_tsv,
                        testing_tsv,
                        str(pred_tsv),
                        k=k,
                        sketch_size=s,
                        ground_truth_tsv=None
                    )

                    # ---- COMPUTE AUC ----
                    mean_auc = plot_roc_curves(
                        predictions_file=str(pred_tsv),
                        ground_truth_file=ground_truth_tsv,
                        output_file=None
                    )

                    writer.writerow([
                        k,
                        s,
                        f"{mean_auc:.6f}",
                        f"{run_stats['total_time']:.3f}",
                        f"{run_stats['peak_ram']:.2f}"
                    ])

                    results.append({
                        "K": k,
                        "sketch_size": s,
                        "mean_auc": mean_auc,
                        "time": run_stats["total_time"],
                        "ram": run_stats["peak_ram"]
                    })

                    if mean_auc > best_auc:
                        best_auc = mean_auc
                        best_parameters = (k, s)
                        
                    pbar.set_postfix({
                        "K": k,
                        "S": s,
                        "AUC": f"{mean_auc:.3f}"
                    })

                    pbar.update(1)

    print(f"\nSaved optimization results to {output_results_tsv}")
    print(
        f"Best AUC = {best_auc:.4f} "
        f"for parameters K={best_parameters[0]}, sketch_size={best_parameters[1]}"
    )

    return results, best_parameters




if __name__ == "__main__":
    training = "data/train0_data.tsv"
    testing = "data/test0_data.tsv"
    ground_truth = "data/test0_ground_truth.tsv"
    
    param_grid = {
        "K": np.arange(6, 40, 2),
        "sketch_size": np.arange(500, 3000, 150)
    }

    optimize_parameters(
        training_tsv=training,
        testing_tsv=testing,
        ground_truth_tsv=ground_truth,
        param_grid=param_grid,
        output_results_tsv="results/optimization_results.tsv"
    )
