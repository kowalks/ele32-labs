from cmath import inf
from matplotlib.pyplot import axis
import numpy as np

from channel import BPSK_AWGN

class Encoder:
    name = 'Abstract encoder'

    def encode(self, word):
        raise NotImplementedError()
    
    def decode(self, word):
        raise NotImplementedError()

    def hamming_weight(self,word):
        return word.sum()

    def hamming_dist(self, u, v):
        k = (u+v) % 2
        return self.hamming_weight(k)

class HammingEncoder(Encoder):
    name = 'Hamming encoder'

    G = np.array([[1,0,0,0,1,1,1],[0,1,0,0,1,0,1],[0,0,1,0,1,1,0],[0,0,0,1,0,1,1]])
    Ht = np.array([[1,1,1],[1,0,1],[1,1,0],[0,1,1],[1,0,0],[0,1,0],[0,0,1]])
    syn_err_dict = {
        '[0 0 0]': np.array([0,0,0,0,0,0,0]),
        '[0 0 1]': np.array([0,0,0,0,0,0,1]),
        '[0 1 0]': np.array([0,0,0,0,0,1,0]),
        '[0 1 1]': np.array([0,0,0,1,0,0,0]),
        '[1 0 0]': np.array([0,0,0,0,1,0,0]),
        '[1 0 1]': np.array([0,1,0,0,0,0,0]),
        '[1 1 0]': np.array([0,0,1,0,0,0,0]),
        '[1 1 1]': np.array([1,0,0,0,0,0,0]),
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

class CyclicEncoder(Encoder):
    name = 'Hamming encoder'

    def __init__(self, G,H,mapped_s):
        self.G = G
        self.H = H
        self.g = self._trim(G[0,:])
        # print(f"g = {self.g}")
        self.h = self._trim(H[0,:])
        # raiz é 1+D^n
        self.raiz = np.zeros(G.shape[1]+1, dtype=int)
        self.raiz[-1] = self.raiz[0] = 1
        # mapped_s é o array de palavras sindromes mapeadas, nesse caso [[1 .... 0]]
        self.mapped_s = mapped_s

    def _trim(self, u):
        return np.trim_zeros(u, 'b')

    def _mod(self, u, m):
        m = self._trim(m)
        u = self._trim(u)
        if len(u) < len(m):
            u = np.pad(u, (0,len(m)-len(u)-1))
            return u

        z = np.zeros(len(u)-len(m), dtype=int)
        k = np.concatenate([z, m])
        u = (k + u) % 2
        u = self._trim(u)
        return self._mod(u, m)

    def _rotate(self, u, m):
        z = np.zeros(1, dtype=int)
        u = np.concatenate([z, u])
        u = self._mod(u, m)
        return np.pad(u, (0,len(m)-len(u)-1))

    def _sum(self,a,b):
        if len(a) < len(b):
            c = b.copy()
            c[:len(a)] += a
        else:
            c = a.copy()
            c[:len(b)] += b
        return c % 2

    def _divide(self,v,g):
        k  = len(v) - len(g)
        if k < 0:
            return np.array([0])

        q = np.zeros(k+1, dtype=int) 
        q[-1] = 1

        zeros = np.zeros(k, dtype=int)
        g_shift = np.concatenate([zeros, g])

        soma = (g_shift + v) % 2
        soma = self._trim(soma)

        return self._sum(q, self._divide(soma,g))
        
    def encode(self, u):
        # u = np.pad(u, (0,G.shape[0]-lenG(u)))
        v = np.dot(u, self.G) % 2
        return v

    def decode(self, v):
        # calcular sindrome
        # da pra colocar n no init
        n = self.G.shape[1]
        s = self._mod(v,self.g)
        s_init = np.copy(s)
        if self.hamming_weight(s) == 0:
            return self._divide(v,self.g)
        
        # print(f"s = {s}")
        count = 0
        while not np.any([np.array_equal(s, ms) for ms in self.mapped_s]):
            s = self._rotate(s,self.g)
            v = self._rotate(v,self.raiz)
            count += 1
            # print(f"count = {count} // s = {s} // s_init = {s_init}")
            if np.array_equal(s,s_init):
                return self._divide(v,self.g)
        # corrigindo primeiro bit
        v[0] = not v[0]
        
        # desrotacionando vetor
        for _ in range(n-count):
            v = self._rotate(v,self.raiz)

        # print(f"final v= {v}")
        return self.decode(v)

class ConvolutionalEncoder(Encoder):
    name = 'Convolutional encoder'

    def __init__(self, n, m, G):
        self.n = n
        self.m = m
        self.G = G

    def _compute_transition(self, bit, M):
        M = np.concatenate([[bit], M])
        v = np.logical_and(self.G, M)
        v = v.sum(axis=1) % 2
        M = M[:-1]
        return v, M


    def encode(self, word):
        M = np.zeros(self.m)
        V = []
        for bit in word:
            v, M = self._compute_transition(bit, M)
            V.append(v)
        V = np.array(V).reshape(-1)
        return V

    def decode(self, word, dist_func=None):
        if dist_func == None:
            dist_func = self.hamming_dist

        word = word.reshape((-1, self.m))
        minDist = {}
        minPath = {}

        for i in range(2**self.m):
            minDist[i] = inf
            minPath[i] = []
        minDist[0] = 0

        for k in range(word.shape[0]):
            bits = np.array(list(range(2**self.m)), dtype='uint8')
            nodes = np.unpackbits(bits).reshape((-1, 8))[:, -self.m:]
            newDist = {}
            newPath = {}
            for node_repr, node in enumerate(nodes):
                v1, child1 = self._compute_transition(0, node)
                v2, child2 = self._compute_transition(1, node)

                repr1 = np.packbits(np.flip(child1), bitorder='little')[0]
                nd = minDist[node_repr] + dist_func(v1, np.array(word[k]))

                if (newDist.get(repr1) and nd < newDist.get(repr1)) or newDist.get(repr1) is None:
                    newDist[repr1] = nd
                    newPath[repr1] = minPath[node_repr] + [0]

                repr2 = np.packbits(np.flip(child2), bitorder='little')[0]
                nd = minDist[node_repr] + dist_func(v2, np.array(word[k]))

                if (newDist.get(repr2) and nd < newDist.get(repr2)) or newDist.get(repr2) is None:
                    newDist[repr2] = nd
                    newPath[repr2] = minPath[node_repr] + [1]
            minDist = newDist
            minPath = newPath

        node = min(minDist, key=minDist.get)
        path = minPath[node]
        return np.array(path)


class ConvEncoderEuclidean(ConvolutionalEncoder):
    name = 'Convolutional encoder with Euclidean Distance'

    def decode(self, word):
        def func(u, v):
            return np.linalg.norm(u-v)**2
        return super().decode(word, dist_func=func)



if __name__ == '__main__':
    n = 3
    m = 3
    G = np.array([[1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]])

    enc = ConvEncoderEuclidean(n, m, G)
    channel = BPSK_AWGN()

    word = np.array([1,0,0,1,1,0,0,0,1,0], dtype=int)

    encoded = enc.encode(word)
    transmitted = channel.transmit(encoded)
    decoded = enc.decode(transmitted)
    
    print(word)
    print(encoded)
    print(transmitted)
    print(decoded)
    