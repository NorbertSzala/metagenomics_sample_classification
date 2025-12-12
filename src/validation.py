
# Norbert: # measure_time

# Kasia: Measure RAM and compare outputs



import time
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
