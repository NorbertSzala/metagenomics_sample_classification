#validation

# Norbert: # measure_time

# Kasia: Measure RAM and compare outputs



import time
import psutil
import os
from typing import Callable, Any

def measure_time(func: Callable, *args, n_units: int | None=None, **kwargs ):
    """
    Measure execution time of given function

    Args:
        func (Callable): function to be measured
        *args: positional arguments for func
        n_units (int | None, optional): Number of processed units, int or None, eg. reads, samples. Defaults to None.
        **kwargs: keywoard arguments to func (keys from dict)
    """
    
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    
    total_time = end - start
    
    # count operation tiem of read, sample etc
    time_per_unit = None
    if n_units is not None and n_units >0:
        time_per_unit = total_time/n_units
        
    return {'result':result, 'total_time': total_time, 'time_per_unit':time_per_unit}

def measure_ram() -> float:
    """
    Get current RAM usage in MB.

    Usage: at some point you are interested in you can call:
    cur_ram =  measure_ram()
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def compare_outputs(predictions: dict[str, str], 
                   ground_truth: dict[str, str]):
    """
    Compare predictions to ground truth.
    
    Args:
        predictions: Dict: [sample IDs, predicted labels]
        ground_truth: Dict: [sample IDs, true labels]
        
    Returns:
        Dict with accuracy metrics
    """
    correct = 0
    total = 0
    
    for sample_id, pred_label in predictions.items():
        if sample_id in ground_truth:
            total += 1
            if pred_label == ground_truth[sample_id]:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


