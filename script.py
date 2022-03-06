import time
import os

import numpy as np

import ml_files.poem_cleaner
import ml_files.poem_collector
import ml_files.poem_modeling_functions as pm

def script():
    main_path = os.getcwd()
    
    #Load poems and text
    print('Loading poems...')
    poem_text = ml_files.poem_cleaner.join_datasets()
    poem_length = ml_files.poem_cleaner.avg_poem_length(poem_text)
    cleaned_poem = ml_files.poem_cleaner.clean_poems(poem_text)

    #Tokenize, generate model, and train
    print('Generating Tokenizer and Model...')
    tokenizer = pm.generate_tokenizer(cleaned_poem)
    my_model = pm.generate_model(len(tokenizer.word_index))


    #Generate train,valid, and test datsets
    print('Generating Datasets...')
    train_ds = pm.generate_dataset(cleaned_poem,tokenizer,perc_start = 0, perc_end = 0.85)
    valid_ds = pm.generate_dataset(cleaned_poem,tokenizer,perc_start = 0.85, perc_end = 0.95)
    test_ds = pm.generate_dataset(cleaned_poem,tokenizer,perc_start = 0.95, perc_end = 1)


    #Train Model
    print('Training...  ')
    my_model, history = pm.train_model(my_model,train_ds,valid_ds,epochs = 10,learning_rate = 5e-3)


    #Evaluate Model
    print('Evaluating.. ')
    result = my_model.evaluate(test_ds)
    print(my_model.metrics_names + '   ' + str(result))


    #Generate new poems!
    print('Generating poems....')
    new_poems = ""

    for tmp in [0.01, 0.1, 1, 2, 10]:
        print(f'Temperature = {tmp}:')
        my_poem = pm.generate_poem(my_model,tokenizer,poem_length = poem_length,temperature = tmp)
        new_poems += f'This poem was generated with a temperature of {tmp}'
        new_poems += my_poem


    print('Saving poems...')
    new_poem_fid = time.strftime( main_path + '/new_poems/poems_%H_%M_%S.txt')
    with open(new_poem_fid,'w') as f:
        f.write(new_poems)
        
if __name__ == '__main__':
    script()