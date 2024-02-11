
import random
import numpy as np
from   numpy.linalg import norm


## Utility functions
# Implementation of cosine similarity, for determining distance between identity attributes
def cosine_similarity(a,b):
    return np.dot(a, b)/(norm(a)*norm(b))   

## Globals
# Define global variable to hold the database, referenced by the db functions

## Database functions
# Initialize new database
def create():
    data = np.ones([64])
    # for i in range(10):
    #     entry = np.array([random.random() for i in range(64)])
    #     data = np.vstack((data, entry))
    #     # Save test identity

    # data = np.delete(data, 0, 0)
    # np.save('faces.npy', data)
    return data

def save(db, file):
    print(f" -- saving database to '{file}'")
    np.save(file, db)
    
# Load the database to memory    
def load(file):
    return  np.load(file)

# Add entry to the database
def add_entry(db, arr):
    return np.vstack((db, arr))

# Query the database    
def query(db, arr):
    distance  = [1-cosine_similarity(arr, e) for e in db]
    min_index = distance.index(min(distance))
    min_value = distance[min_index]



    return (min_index, min_value)

# Test function to see that everything works
def load_and_test():
    load()

    # Pick identity and slight shift its values to assert we can still find the correct one
    identity = [i + 0.03*random.random() for i in db[7]]
    (i, v) = query(identity)

    print(db)
    
    print(i, v)
    assert(i == 7)
