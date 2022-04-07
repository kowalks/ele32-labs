import numpy as np

class Encoder:
    name = 'Abstract encoder'

    def encode(self, word):
        raise NotImplementedError()
    
    def decode(self, word):
        raise NotImplementedError()
    

class HammingEncoder(Encoder):
    name = 'Hamming encoder'

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

class NaiveEncoder(Encoder):
    name = 'Naive encoder'

    def encode(self, u):
        return u
    
    def decode(self, r):
        return r

class AlternativeEncoder(Encoder):
    name = 'Alternative encoder'

    G = np.array([[1,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
                  [0,1,0,0,0,0,0,0,0,0,0,0,1,1,1],
                  [0,0,1,0,0,0,0,0,0,0,0,1,0,1,1],
                  [0,0,0,1,0,0,0,0,0,0,0,1,1,0,1],
                  [0,0,0,0,1,0,0,0,0,0,0,1,1,1,0],
                  [0,0,0,0,0,1,0,0,0,0,0,0,0,1,1],
                  [0,0,0,0,0,0,1,0,0,0,0,0,1,0,1],
                  [0,0,0,0,0,0,0,1,0,0,0,0,1,1,0],
                  [0,0,0,0,0,0,0,0,1,0,0,1,0,1,0],
                  [0,0,0,0,0,0,0,0,0,1,0,1,1,0,0],
                  [0,0,0,0,0,0,0,0,0,0,1,1,0,0,1]])
    Ht = np.concatenate([G[:,-4:], np.eye(4, dtype=int)],axis=0)
    syn_err_dict = {
        '[1 1 1 1]': np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        '[0 1 1 1]': np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        '[1 0 1 1]': np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]),
        '[1 1 0 1]': np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),
        '[1 1 1 0]': np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]),
        '[0 0 1 1]': np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]),
        '[0 1 0 1]': np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
        '[0 1 1 0]': np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]),
        '[1 0 1 0]': np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]),
        '[1 1 0 0]': np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]),
        '[1 0 0 1]': np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
        '[1 0 0 0]': np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]),
        '[0 1 0 0]': np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]),
        '[0 0 1 0]': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]),
        '[0 0 0 1]': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]),
        '[0 0 0 0]': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
    }

    def encode(self, u):
        v = np.dot(u, self.G) % 2
        return v

    def decode(self, r):
        s = np.dot(r, self.Ht) % 2
        e = self.syn_err_dict[np.array2string(s)]
        c = (r + e) % 2
        return c[:11]