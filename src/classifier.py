# Norbert - load data + train model
# classifier.py

import argparse

from compute_sketch import compute_sketch
from model import train_model
from utils_io import load_metadata
from validation import measure_time


####################################
########## SET PARAMETERS ##########
####################################

K = 21
SKETCH_SIZE = 2000


####################################
############## MAIN ################
####################################

model = train_model(training_data, K, SKETCH_SIZE)


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
        testing_tsv (str): path to testing tsv
        output_tsv (str): path to output
        k (int, optional): Kmer size. Defaults to K (21).
        sketch_size (int, optional): number of sketches in single probe. Defaults to SKETCH_SIZE.
    """
    
    # ~~~~~ 1. load training data ~~~~~
    training_data = load_metadata(training_tsv)
    
    
    # ~~~~~ 1. build training model ~~~~~
    # measure time
    train_stats = measure_time(
        train_model, training_data, k, sketch_size, n_units=len(training_data)
    )
    
    model = train_stats["results"]
    

    # NOTE:
    # model: dict[class_name -> list[sketches]]
    # No normalization here (by design)
    
    # ~~~~~ 3. load testing data ~~~~~
    testing_data = load_metadata(testing_tsv)
    # testing_data contains only fasta_file (class_geo_loc may be none)
    
    # ~~~~~ 4. classification (TODO) ~~~~~
    # TODO:
    # for each test sample:
    #   - compute sketch
    #   - compare to model
    #   - compute score per class
    #
    # results = classify_all(testing_meta, model, k, sketch_size)

    raise NotImplementedError(
        "Classification step not implemented yet "
        "(classify_sample / classify_all missing)"
    )

    # ~~~~~ 5. write output (TODO) ~~~~~
    # write_output(results, output_tsv)



# TODO: write if __name__ = .. with argument parser