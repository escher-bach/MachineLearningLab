import numpy as np
import random
#take in two numpy arrays x and y then split them on the basis of the test train split
def train_test_split(X, y, test_size=0.2, random_state=None):
    #concatenate X and y for shuffling
    df = np.concatenate((X, y.reshape(-1, 1)), axis=1) #reshape y to be 2D
    if(random_state):
        np.random.seed(random_state) #set seed for reproducibility
    np.random.shuffle(df) #shuffle the data
    #now we find the point at which to split
    split_index = int(len(df) * (1 - test_size))
    train = df[:split_index]
    test = df[split_index:]
    #now we take out the x and y 
    x_train = train[:, :-1]
    y_train = train[:, -1].reshape(-1, 1)
    x_test = test[:, :-1]
    y_test = test[:, -1].reshape(-1, 1)
    return x_train, y_train, x_test, y_test