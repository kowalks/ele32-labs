import random
import numpy as np

class Encoder:
    def encode(self, word):
        raise NotImplementedError()
    
    def decode(self, word):
        raise NotImplementedError()
    

class Hamming(Encoder):
    G = np.array([[1,0,0,0,1,0,1],[0,1,0,0,1,1,0],[0,0,1,0,1,1,1],[0,0,0,1,0,1,1]])
    Ht = np.array([[1,0,1],[1,1,0],[1,1,1],[0,1,1],[1,0,0],[0,1,0],[0,0,1]])
    syn_err_dict = {
        '[0 0 0]': np.array([0,0,0,0,0,0,0]),
        '[0 0 1]': np.array([0,0,0,0,0,0,1]),
        '[0 1 0]': np.array([0,0,0,0,0,1,0]),
        '[0 1 1]': np.array([0,0,0,1,0,0,0]),
        '[1 0 0]': np.array([0,0,0,0,1,0,0]),
        '[1 0 1]': np.array([1,0,0,0,0,0,0]),
        '[1 1 0]': np.array([0,1,0,0,0,0,0]),
        '[1 1 1]': np.array([0,0,1,0,0,0,0]),
    }

    def encode(self, u):
        v = np.dot(u, self.G) % 2
        return v

    def decode(self, r):
        s = np.dot(r, self.Ht) % 2
        e = self.syn_err_dict[np.array2string(s)]
        c = (r + e) % 2
        return c[:4]


class BSC:
    def __init__(self, encoder, k, l):
        self.encoder
        self.k = k
        self.l = l
        self.words = []

    def generate_bits(self, seed=2022):
        for i in range(self.l):
            word = random.choices([0,1], k=self.k)
            self.words.append(word)

    def transmit(self, prob=0.1):
        pass

        
