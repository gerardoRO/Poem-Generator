import tensorflow as tf
from tensorflow import keras

from . import poem_collector
from . import poem_cleaner

import numpy as np

import time
import os

fid_path = os.path.dirname(__file__)

def generate_tokenizer(text,char_level = True):
    """Generate keras tokenizer based on our poems' text

    Args:
        text (str): Poems' text
        char_level (bool, optional): character-based tokenizer. Defaults to True.

    Returns:
        tokenizer: keras tokenizer
    """
    tokenizer = keras.preprocessing.text.Tokenizer(char_level = char_level)
    tokenizer.fit_on_texts(text)
    
    return tokenizer  

def generate_dataset(text,tokenizer,perc_start = 0, perc_end = 0.85,win_size = 50):
    """Generate dataset object based on the poems text.

    Args:
        text (str): poem text to generate dataset
        tokenizer (keras tokenizer): previously fit tokenizer for the poem
        train_size (float, optional): size of the train sized. Defaults to 0.85.
        win_size (int, optional): size of the characer window for each item in database. Defaults to 50.

    Returns:
        (dataset, train_indx): tuple with the tf.Dataset object and the index designating when the train data set changes.
    """
    tot_depth = len(tokenizer.word_index)
    start_indx = int( perc_start * tokenizer.document_count)
    end_indx = int( (perc_start + perc_end) * tokenizer.document_count)
    
   
    #encode text into integers
    [encoded] = np.array(tokenizer.texts_to_sequences([text])) - 1
    dataset = tf.data.Dataset.from_tensor_slices(encoded[start_indx:end_indx])

    
    #window dataset
    dataset = dataset.window(win_size+1,shift = 1,drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(win_size+1))
    
    #batch and split into features and labels
    dataset = dataset.shuffle(10000).batch(32) 
    dataset = dataset.map(lambda this_sample: (this_sample[:,:-1],this_sample[:,1:]))
    
    #generate one_hot_encoding
    dataset = dataset.map(lambda x,y: (tf.one_hot(x,depth = tot_depth),y))  
    dataset = dataset.prefetch(1)
    return dataset
    
def generate_model(num_tokens):
    """Create 2 layer GRU model for generating poems.

    Args:
        num_tokens (int): size of tokenizer units. Corresponds to diversity of model classes.

    Returns:
        keras model: Trained keras model.
    """
    model = keras.models.Sequential([
        keras.layers.GRU(128,return_sequences = True, dropout = 0.2, input_shape = [None,num_tokens]),
        keras.layers.GRU(128,return_sequences = True, dropout = 0.2),
        keras.layers.TimeDistributed(keras.layers.Dense(num_tokens,activation = 'softmax'))
    ])
    
    
    
    return model

def train_model(model,dataset_train,dataset_valid,epochs=1,learning_rate = 1e-3):
    """Train keras model.

    Args:
        model (keras model): keras model for poetry generation.
        dataset_train (tensorflow dataset): tensorflow dataset to train the model with extracted poems.
        dataset_valid (tensorflow dataset): tensorflow dataset to validate the model.
        epochs (int, optional): number of epochs to train model. Defaults to 1.
    """
    model_name = time.strftime(fid_path + '/../models/poem_model_%Y_%m_%d_%H_%M_%S.h5')
    log_file = time.strftime(fid_path + '/../logs/log_%Y_%m_%d_%H_%M_%S')
    
    early_cb = keras.callbacks.EarlyStopping(patience = 5,restore_best_weights = True)
    save_cb = keras.callbacks.ModelCheckpoint(model_name,save_best_only = True)
    nan_cb = keras.callbacks.TerminateOnNaN()
    tb_cb = keras.callbacks.TensorBoard(log_file)
    
    schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 5e-3,decay_steps = 1000, decay_rate = .8)
    lr_cb = keras.callbacks.LearningRateScheduler(schedule, verbose = 1)
    
    my_opt = keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss = 'sparse_categorical_crossentropy',optimizer = my_opt)
    history = model.fit(dataset_train,epochs = epochs,validation_data = dataset_valid, callbacks =[early_cb,save_cb,tb_cb,nan_cb,lr_cb])
    
    model.save('tmpy.h5')
    model.save(fid_path + '/../models/fin_model_%Y_%m_%d_%H_%M_%S.h5')
    
    return model,history

def load_model(model_name):
    """ Load previously generated model

    Args:
        model_name (string): keras model
    """
    keras.models.load_model(fid_path + '/../models/' + model_name)
    
    return

def generate_poem(model,tokenizer,poem_length=10,temperature = 1,starting_string = 'love'):
    """Generate a poem based on the inputted model

    Args:
        model (keras model): Keras model
        tokenizer (tensorflow tokenizer): Tokenizer used to encode for the model
        poem_length (int, optional): Total length of the poem. Defaults to 10.
        temperature (int, optional): Temperature to adjust the logits by. Defaults to 1.
        starting_string (str, optional): Starting string for the poem. Defaults to 'Love'.

    Returns:
        str: Created poem
    """
    new_poem = starting_string
    while len(new_poem) < poem_length:
        new_token = np.array(tokenizer.texts_to_sequences([new_poem])) - 1 
        Xnew = tf.one_hot(new_token,depth = len(tokenizer.word_index))
        
        y_pred = model(Xnew)[0,-1,:] #predict based on the texts 

        rescaled_logits = tf.math.log(y_pred) / temperature #generate probits and adjust by temperature

        pred_char = tf.random.categorical(tf.reshape(rescaled_logits,[1,len(rescaled_logits)]),num_samples = 1) + 1 
        new_poem += tokenizer.sequences_to_texts(pred_char.numpy())[0] #append new text to the generated_poem 
        
    return new_poem