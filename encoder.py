from matplotlib.pyplot import axis
import numpy as np

class Encoder:
    name = 'Abstract encoder'

    def encode(self, word):
        raise NotImplementedError()
    
    def decode(self, word):
        raise NotImplementedError()
    def hamming_weight(self,word):
        return word.sum()

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

    def __init__(self, G,H):
        self.G = G
        self.H = H
        self.g = self._trim(G[0,:])
        self.h = self._trim(H[0,:])
        # raiz é 1+D^n
        self.raiz = np.zeros(G.shape[1]+1, dtype=int)
        self.raiz[-1] = self.raiz[0] = 1
        # mapped_s é o array de palavras sindromes mapeadas, nesse caso [[1 .... 0]]
        k = np.zeros(self.g.shape[0]-1)
        k[0] = 1
        self.mapped_s = [np.array(k, dtype=int)]
        # print(f"g = {self.g}")
        # print(f"r = {self.raiz}")
        # print(f"ms = {self.mapped_s}")

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
            return v

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
        
        count = 0
        while not np.any([np.array_equal(s, ms) for ms in self.mapped_s]):
            # print(f"s = {s} // mapped_s = {self.mapped_s}")
            # print(f"array: {[np.array_equal(s, ms) for ms in self.mapped_s]}")
            s = self._rotate(s,self.g)
            v = self._rotate(v,self.raiz)
            count += 1
            # print(f"count = {count} // s = {s} // s_init = {s_init}")
            if np.array_equal(s,s_init):
                print("ERROR -> erro não pode ser mapeado")
                break
        # corrigindo primeiro bit
        v[0] = not v[0]
        
        # desrotacionando vetor
        for _ in range(n-count):
            v = self._rotate(v,self.raiz)

        # print(f"final v= {v}")
        return self.decode(v)


if __name__ == '__main__':
    import numpy as np

    G = np.genfromtxt('lab2_values/g10_6.csv', delimiter=',', dtype=int)
    H = np.genfromtxt('lab2_values/h10_6.csv', delimiter=',', dtype=int)
    ce = CyclicEncoder(G,H)
    
    u = np.array([1,0,0,0,0,0])
    print(f"u = {u}")

    v = ce.encode(u)
    print(f"v = {v}")

    e = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0])
    ve = (v + e) % 2
    print(f"ve = {ve}")

    uf = ce.decode(ve)
    print(f"uf = {uf}")