# main.py

import sys
import numpy as np

from compute_sketch import compute_sketch
from model import train_model
from utils_io import load_metadata, write_output_tsv
from validation import measure_time, measure_ram
from similarity import classify_all
from plotting import plot_roc_curves
from pathlib import Path
from logs import RunLogger

####################################
########## SET PARAMETERS ##########
####################################

K = 21
SKETCH_SIZE = 2000


####################################
############## MAIN ################
####################################


def main(training_tsv: str, testing_tsv: str, output_tsv: str, k: int = K, sketch_size:int = SKETCH_SIZE, ground_truth_tsv: str | None=None):
    """
    Main pipeline of the programme
    1. load training data
    2. build reference model (representative sketches per class)
    3. load testing metadata
    4. classivy test samples    # to implement
    5.. write output tsv        # to implement

    Args:
        training_tsv (str): path to training tsv
        testing_tsv (str): path to  testing tsv
        output_tsv (str): path to output
        k (int, optional): Kmer size. Defaults to K (21).
        sketch_size (int, optional): number of sketches in single probe. Defaults to SKETCH_SIZE.
        ground_truth_tsv (str or None): Optional parameter, default is None. Path to tsv file with provided proper solutions to make plots AUC
    """
    print('Running script')
    
    
    stats = {}
        
    # ~~~~~ 1. load training data ~~~~~
    training_data = load_metadata(training_tsv)
    
    # ~~~~~ steps 2. - 5. with measuring data and time of execution ~~~~~
    # ~~~~~ 2. load testing data ~~~~~
    testing_data = load_metadata(testing_tsv)
    
    # n_units is number of testing probes/units
    with RunLogger(n_units=len(testing_data)) as logger:
        # ~~~~~ 3. build training model ~~~~~
        model_data = train_model(training_data, k, sketch_size)
        logger.update_peak_ram()
    
    
        # ~~~~~ 4. compute test sketches ~~~~~
        test_sketchs = {}
    
        for entry in testing_data:
            fasta_path = entry.fasta_file
            sketch = compute_sketch(fasta_path, k, sketch_size)
            sketch = sorted([-h for h in sketch]) # implementation of min heap

            test_sketchs[str(fasta_path)] = {"sketch": np.array(sketch, dtype=np.uint64)}
        
            logger.update_peak_ram()
    
    
        # ~~~~~ 5. classification ~~~~~~
        scores = classify_all(test_sketchs, model_data)
        

        # ~~~~~ 6. write output  ~~~~~
        write_output_tsv(scores, output_tsv)
        print(f'Output succesfully safed to: {output_tsv}')    
    
    # ~~~~~ optional: plotting ROC curves ~~~~~
    if ground_truth_tsv is not None:
        plot_path = Path(output_tsv).with_suffix(".roc.png")

        plot_roc_curves(
            predictions_file=output_tsv,
            ground_truth_file=ground_truth_tsv,
            output_file=str(plot_path)
        )

    
    return stats
    
    
    
'''
Nie rozumiem po co dawać w tej funkcji reverse complement? Dlaczego uważasz że cała ta funkcja jest tak wolna? Jak ją usprawnić?
def optimize_parameters(training_tsv:str, testing_tsv:str, param_grid:dict):
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
    
    for k in param_grid['K']:
        for s in param_grid['sketch_size']:
            stats = main(training_tsv, testing_tsv, 'tmp.tsv', k, s)
            
            
            results.append({
                'k': k,
                'sketch_size': s,
                'train_time': stats['train']['total_time']
            })
            
    return results
    
    
    
    
2. Jak polecasz naprawić jaccarda?

3. Dodaj odporność na różne nagłówki, niech funkcja sprawdza czy dane rzeczywiście mają ten nagłówek



'''