
import random
import numpy as np
from numpy.linalg import norm

def cosine_similarity(a,b):
    return np.dot(a, b)/(norm(a)*norm(b))   

# Define global variable to hold the database
db = {}

def create():
    data = np.array([1.0, 1.0, 1.0])
    for i in range(10):
        entry = np.array([random.random() for i in range(3)])
        data = np.vstack((data, entry))
        # Save test identity
        np.save('faces.npy', data)

# Load the database to memory    
def load():
    global db
    db = np.load('faces.npy')

# Add entry to the database
def add_entry(arr):
    global db
    db = np.vstack((db, arr))

# Query the database    
def query(arr):
    distance = [1-cosine_similarity(arr, e) for e in db]
    min_index = distance.index(min(distance))
    min_value = distance[min_index]

    return (min_index, min_value)

def load_and_test():
    load()

    identity = [i + 0.03*random.random() for i in db[7]]
    (i, v) = query(identity)
   
    print(i, v)
    assert(i == 7)

