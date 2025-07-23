import random

def randomize_dict_items(input_dict, seed=None):
    """
    Create a new dictionary with the same items as input_dict but in random order.
    
    Parameters
    ----------
    input_dict : dict
        Dictionary to randomize
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        New dictionary with randomized order of items
    """
    if seed is not None:
        random.seed(seed)
    
    items = list(input_dict.items())
    random.shuffle(items)
    return dict(items) 