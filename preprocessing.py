import numpy as np
import pandas as pd
import os

'''
Utility method that will convert the original dataset to numpy arrays.
Removes any instances of black boxes.
Normalizes the images by subtracting the mean and dividing by stdv.
Saves the arrays as .npy files with the labels and images separate.
'''
def preprocess():
    DATA_DIR="datasets"
    RAW_DATA="raw_images.csv"

    #Read in raw data as numpy array
    raw_data = (pd.read_csv(os.path.join(DATA_DIR, RAW_DATA))).to_numpy()
    
    #Delete all instances of black boxes
    raw_filtered = np.copy(raw_data)
    num_black_boxes = 0
    for i in range(raw_data.shape[0]):
        if np.std(np.fromstring(raw_data[i][2], dtype=int, sep=" ")) == 0:
            raw_filtered = np.delete(raw_filtered, obj=(i - num_black_boxes), axis=0)
            num_black_boxes += 1

    #Create arrays and separate labels from images
    labels = raw_filtered[:,0]
    images = np.empty((raw_filtered.shape[0], 2304))

    #Normalize the images & store them in array
    for i in range(images.shape[0]):            
        images[i] = np.fromstring(raw_filtered[i][2], dtype=int, sep=" ")
        images[i] = (images[i] - np.mean(images[i]))/(np.std(images[i]))
    
    #Save images and labels to their respective files
    eighty_percent = np.ceil((.8 * images.shape[0])).astype(int)
    ten_percent = np.ceil((.1 * images.shape[0])).astype(int)
    np.save(os.path.join(DATA_DIR, "train_images"), images[:eighty_percent])
    np.save(os.path.join(DATA_DIR, "train_labels"), labels[:eighty_percent])
    np.save(os.path.join(DATA_DIR, "dev_images"), images[eighty_percent:eighty_percent + ten_percent])
    np.save(os.path.join(DATA_DIR, "dev_labels"), labels[eighty_percent:eighty_percent + ten_percent])
    np.save(os.path.join(DATA_DIR, "test_images"), images[eighty_percent + ten_percent:])
    np.save(os.path.join(DATA_DIR, "test_labels"), labels[eighty_percent + ten_percent:])

preprocess()
