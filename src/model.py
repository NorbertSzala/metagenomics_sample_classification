# Norbert
# model.py


'''
Function reads training data and counts sketch for every probe. Next, it grouping according to class (place)

Medium-main goal is to build referential model (set of sketches), used by classicier.py in the next steps.
That model represents equally each class and is very very smaller than original data,- fingerprint-> faster computation, 

'''

from compute_sketch import compute_sketch


def train_model(metadata:list, k:int=21, sketch_size:int=2000)-> dict:
    """
    Build a classification model as mapping:
        class_name -> list of MinHash sketches for that class
    
    Args:
        metadata (list): list of objects with attributes: fasta_file and class_name (both str)
        k (int, optional): Kmer length Defaults to 21.
        sketch_size (int, optional): sketch size. Defaults to 2000.

    Returns:
        dict[str, list[lsit[int]]]
    """
    
    class_to_sketches = {}
    for entry in metadata:
        cls = entry.class_name
        fasta_path = entry.fasta_file
        
        #compute sketch for the dataset
        sketch = compute_sketch(fasta_path, k, sketch_size)
        
        # initialise class if needed
        if cls not in class_to_sketches:
            class_to_sketches[cls] = []
            
        class_to_sketches[cls].append(sketch)
        
    return class_to_sketches
