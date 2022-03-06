import bs4 
import requests
import pickle
import time
import os

fid_path = os.path.dirname(__file__)   

def find_poem():
    """Find random poem from poetry.com

    Returns:
        tuple: first value is poem url, second value is unprocessed poem text
    """
    
    random_url = 'https://www.poetry.com/random.php'

    poem = requests.get(random_url)
    soup = bs4.BeautifulSoup(poem.text,'html.parser')    

    poem_url = poem.url
    poem_txt = soup.find(id='disp-quote-body').text 
    
    return (poem_url,poem_txt)

def generate_dataset(num_poems = 1,save = True):    
    """Generate a dataset of random poems from poetry.com

    Args:
        num_poems (int, optional): number of poems in the dataset. Defaults to 1.
        save (bool, optional): save poems and poem urls into pickle file. Defaults to True.

    Returns:
        tuple: list of all poem urls, string containing all the poems, separated by ___NEW_POEM___
    """
    poem_urls = []
    all_poem_text = ''
    for _ in range(num_poems):
        poem_url,poem_text = find_poem()
        poem_urls.append(poem_url)
        all_poem_text += poem_text
        all_poem_text += '___NEW_POEM___'  #set a delimiter that we can remove later
        
    if save:
        with open(time.strftime(fid_path + '/../datasets/poemDataset_%H_%M_%S'),'wb') as f:
            pickle.dump((poem_urls,all_poem_text),f)
    
    return (poem_urls,all_poem_text)
 
def load_dataset(fid = 'poemDataset_14_27_54'):
    """Load Dataset

    Args:
        fid (str, optional): Filename to read. Defaults to 'file_14_27_54'.

    Returns:
        string: Poem text
    """
    _,poem_text = pickle.load(open(fid_path + '/../datasets/' + fid,'rb'))
    return poem_text

if __name__ == '__main___':
     generate_dataset(num_poems = 10)
    
    
    