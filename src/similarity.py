import numpy as np

def jaccard_similarity(sketch1:np.ndarray, sketch2: np.ndarray):
    '''
    Calcucate jaccard simularity between two sketches, by calculating 
    the common part and deviding by length of the sketch(same for two)
    '''

    if len(sketch1) != len(sketch2):
        raise ValueError("Sketches must have the same size")

    matches = np.sum(sketch1 == sketch2)

    print(matches / len(sketch1))
    return matches / len(sketch1)


def classify(test_sketch: np.ndarray, training_sketches: dict):
    """
    Classify test sketch by comparing to training sketches
    """
    similarities = []
    
    for label, train_sketch in training_sketches.items():
        similarity = jaccard_similarity(test_sketch, train_sketch)
        similarities.append((label, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities

def classify_all(test_sketchs: np.ndarray, training_sketches: dict):
    """
    Calculate similarity scores for all classes.
    """
    scores = {}

    for fasta_name, values in test_sketchs.items():
        similarity = classify(values['sketch'], training_sketches)
        scores[fasta_name] = similarity

    
    return scores
