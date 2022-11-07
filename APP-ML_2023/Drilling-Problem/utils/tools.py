# version 2.1 du 21 mai 2022
# version 2.2 du 9 juin : now the confusion matrix is build in the notebook....

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
    
def plot_loss_accuracy(history):
    '''Plot training & validation loss & accuracy values, giving an argument
       'history' of type 'tensorflow.python.keras.callbacks.History'. '''
    
    plt.figure(figsize=(15,5))
    ax1 = plt.subplot(1,2,1)
    if history.history.get('accuracy'):
        ax1.plot(np.array(history.epoch)+1, history.history['accuracy'], 'o-',label='train')
    if history.history.get('val_accuracy'):
        ax1.plot(np.array(history.epoch)+1, history.history['val_accuracy'], 'o-', label='val')
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch') 
    ax1.grid()
    ax1.legend(loc='best')
    
    # Plot training & validation loss values
    ax2 = plt.subplot(1,2,2)
    if history.history.get('loss'):
        ax2.plot(np.array(history.epoch)+1, history.history['loss'], 'o-', label='train')
    if history.history.get('val_loss'):
        ax2.plot(np.array(history.epoch)+1, history.history['val_loss'], 'o-',  label='val')
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='best')
    ax2.grid()
    plt.show()

def plot_images(image_array, r, L, C):
    '''Plot the images of image_array on a grid L x C, starting at
       rank r'''
    plt.figure(figsize=(C,L))
    for i in range(L*C):
        plt.subplot(L, C, i+1)
        plt.imshow(image_array[r+i], cmap='gray')
        plt.xticks([]); plt.yticks([])
