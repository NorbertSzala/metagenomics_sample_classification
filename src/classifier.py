# Norbert - load data + train model
# classifier.py

import argparse
import sys
import numpy as np

from compute_sketch import compute_sketch
from model import train_model
from utils_io import load_metadata, write_output_tsv
from validation import measure_time, measure_ram
from similarity import classify_all

####################################
########## SET PARAMETERS ##########
####################################

K = 21
SKETCH_SIZE = 2000


####################################
############## MAIN ################
####################################


def main(training_tsv: str, testing_tsv: str, output_tsv: str, k: int = K, sketch_size:int = SKETCH_SIZE):
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
    """
    
    stats = {}

    initial_ram = measure_ram()
    print(f"Initial RAM usage: {initial_ram:.2f} MB")
    
    # ~~~~~ 1. load training data ~~~~~
    training_data = load_metadata(training_tsv)
    # stats['load_train'] = measure_time(load_metadata, training_tsv)
    # ~~~~~ 2. build training model ~~~~~
    model_data = train_model(training_data, k, sketch_size)

    # measure time
    # model_stats = measure_time(
    #     train_model, training_data, k, sketch_size, n_units=len(training_data)
    # )
    # stats['model_data'] = model_stats
    # print(f"Training time: {model_stats['total_time']:.2f}s")
    # print(f"Time per sample: {model_stats['time_per_unit']:.4f}s")
    
    current_ram = measure_ram()
    print(f"RAM usage after training: {current_ram:.2f} MB")
    
    # ~~~~~ 3. load testing data ~~~~~
    testing_data = load_metadata(testing_tsv)

    test_sketchs = {}

    test_stats = measure_time(
        load_metadata, testing_tsv
    )
 
    # ~~~~~ 4. classification  ~~~~~

    for entry in testing_data:
        fasta_path = entry.fasta_file
        sketch = compute_sketch(fasta_path, k, sketch_size)
        sketch = sorted([-h for h in sketch])

        test_sketchs[fasta_path] = {"sketch": np.array(sketch, dtype=np.uint64)}
        
    scores = classify_all(test_sketchs, model_data)

    # results, classify_time = measure_time(
    #     classify_all, test_sketchs, model_data)

    # ~~~~~ 5. write output  ~~~~~
    write_output_tsv(scores, output_tsv)

    return stats


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("training_tsv")
    parser.add_argument("testing_tsv")
    parser.add_argument('output_tsv')
    parser.add_argument('--k', type=int, default = K) #optionally, delete it later. Just for tests
    parser.add_argument('--sketch_size', type=int, default=SKETCH_SIZE) #optionally, delete it later. 
    
    args = parser.parse_args()
    
    main(
        args.training_tsv,
        args.testing_tsv,
        args.output_tsv,
        k=args.k,
        sketch_size=args.sketch_size
    )