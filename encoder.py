from cmath import inf
from tqdm import trange
import numpy as np

from channel import BPSK_AWGN, BSC, IdealChannel

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
        self.Gt = np.packbits(G, axis=1, bitorder='little').reshape(-1)
        self.nodes = (1 << self.m)

    def _compute_transition(self, bit, M):
        M = np.concatenate([[bit], M])
        v = np.logical_and(self.G, M)
        v = v.sum(axis=1) % 2
        M = M[:-1]
        return v, M

    def _transaction(self, bit, M):
        M = (M<<1) + bit
        v = self.Gt & M
        v = np.array(list(map(lambda n: sum(map(int, f"{n:b}")) % 2, v)))
        M = M % self.nodes
        return v, M

    def _bit_from_transaction(self, parent, child):
        parent = (parent<<1) % self.nodes
        if parent == child:
            return 0
        return 1

    def encode(self, word):
        M = 0
        V = []
        for bit in word:
            v, M = self._transaction(bit, M)
            V.append(v)
        V = np.array(V).reshape(-1)
        return V

    def decode(self, word, dist_func=None):
        if dist_func == None:
            dist_func = self.hamming_dist

        word = word.reshape((-1, self.n))
        minDist = np.ones((self.nodes, word.shape[0]+1), int)*inf
        minParent = np.zeros((self.nodes, word.shape[0]), int)

        # minPath[i][0] = []
        minDist[0][0] = 0

        # bits = np.array(list(range(2**self.m)), dtype='uint8')
        for k in trange(word.shape[0]):
            for node in range(self.nodes):
                v1, child1 = self._transaction(0, node)
                nd = minDist[node][k] + dist_func(v1, np.array(word[k]))
                if nd < minDist[child1][k+1]:
                    minDist[child1][k+1] = nd
                    minParent[child1][k] = node

                v2, child2 = self._transaction(1, node)
                nd = minDist[node][k] + dist_func(v2, np.array(word[k]))
                if nd < minDist[child2][k+1]:
                    minDist[child2][k+1] = nd
                    minParent[child2][k] = node

        node = np.argmin(minDist[:,word.shape[0]])
        path = []
        for i in reversed(range(word.shape[0])):
            parent = minParent[node][i]
            bit = self._bit_from_transaction(parent, node)
            node = parent
            path.append(bit)

        path = list(reversed(path))
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

    # enc = ConvEncoderEuclidean(n, m, G)
    # channel = BPSK_AWGN()

    enc = ConvolutionalEncoder(n, m, G)
    channel = BSC(0.1)

    word = np.array([1,0,0,1,1,0,0,0,1,0], dtype=int)

    encoded = enc.encode(word)
    transmitted = channel.transmit(encoded)
    decoded = enc.decode(transmitted)
    
    print(word)
    print(encoded)
    print(transmitted)
    print(decoded)
    