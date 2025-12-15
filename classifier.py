import sys
from pathlib import Path
import argparse

# Add path to folder with all needed modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import main


# Default parameters
K = 21
SKETCH_SIZE = 2000


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("training_tsv")
    parser.add_argument("testing_tsv")
    parser.add_argument('output_tsv')
    parser.add_argument('--k', type=int, default = K) #optionally, delete it later. Just for tests
    parser.add_argument('--sketch_size', type=int, default=SKETCH_SIZE) #optionally, delete it later.   
    parser.add_argument('--ground_truth_tsv', type=str, default=None)
    args = parser.parse_args()
    
    main(
        args.training_tsv,
        args.testing_tsv,
        args.output_tsv,
        k=args.k,
        sketch_size=args.sketch_size,
        ground_truth_tsv = args.ground_truth_tsv
    )