import pickle
import os.path

def safe_load(filename, callback):
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
    data = callback()
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    return data