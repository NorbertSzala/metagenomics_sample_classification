# Norbert - find the best parameters to model

from classifier import main

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