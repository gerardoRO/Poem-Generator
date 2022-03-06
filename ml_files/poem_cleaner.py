import numpy as np
from scipy import stats
from . import poem_collector
import os
import re

fid_path = os.path.dirname(__file__)   

def join_datasets():
    """Join all poem datasets created into a single text file

    Returns:
        string: return all the texts from the datasets merged together
    """
    datasets = [fid for fid in os.listdir(fid_path + '/../datasets/') if fid.find('poemDataset_')>=0]
    all_txt = ''
    for fid in datasets:
        poem_text = poem_collector.load_dataset(fid)
        all_txt += poem_text

    return all_txt

def avg_poem_length(poem_text,method = 'mode'):
    """Return the average length of the poems

    Args:
        poem_text (string): Text of the poem
        method (str, optional): How to estimate poem length (options: 'median','average','mode'). Defaults to 'mode'.

    Returns:
        int: Average length of the poems in the dataset
    """
    delim = '___NEW_POEM___'
    delim_len = len(delim)
    
    poem_start_ids =[x.start() for x in re.finditer(delim,poem_text)]
    poem_start_ids.insert(0,0)
    
    poem_length = np.diff(np.array(poem_start_ids)) - delim_len
    
    if method == 'mode':
        poem_length = stats.mode(poem_length,axis = None)[0][0]
    elif method == 'median':
        poem_length = np.median(poem_length)
    elif method == 'average':
        poem_length = np.mean(poem_length)
    
    return poem_length

def clean_poems(poem_text,extra_remove = []):
    """Clean up the text of the poems, by removing the delimiters and normalizing cases.

    Args:
        poem_text (string): poem text to clean up.
        extra_remove (list, optional): extra delimiters to remove. Defaults to [].
        
    Returns:
        string: cleaned up poem text.
    """
    
    poem_text = poem_text.lower()
    
    to_remove = ["___NEW_POEM___","\r","\n","\\","!","?",
                 'í','á','-','ñ','©','é','ó','¿','¡','\t',
                 '~','î',"*","(",")","/"]
    to_remove.extend(extra_remove)
    
    for tag in to_remove:
        poem_text = poem_text.replace(tag,'')
        
    for extra_spacing in set(re.findall(' {3,100}',poem_text)):
        poem_text = poem_text.replace(extra_spacing,'  ')
    
    return poem_text