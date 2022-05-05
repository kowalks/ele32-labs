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
        self.raiz = np.zeros(G.shape[1])
        self.raiz[-1] = 1
        # mapped_s é o array de palavras sindromes mapeadas, nesse caso [[1 .... 0]]
        k = np.zeros(G.shape[0])
        k[0] = 1
        self.mapped_s = np.array([k])
    def _trim(self, u):
        return np.trim_zeros(u, 'b')

    def _mod(self, u, m):
        m = self._trim(m)
        u = self._trim(u)
        if len(u) < len(m):
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
        # print("g ", g)
        # print("v ", v)
        # v = self._trim(v)
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
        v_r = np.copy(v)
        if self.hamming_weight(s) == 0:
            return self._divide(v,self.g)
        count = 0
        print(s)
        while s not in self.mapped_s:
            s = self._rotate(s,self.g)
            v_r = self._rotate(v,self.raiz)
            count += 1
            print(s)
            if s == s_init:
                print("ERROR -> erro não pode ser mapeado")
                break
            if count >= n:
                break
        # corrigindo primeiro bit
        v_r[0] = 1
        # desrotacionando vetor
        
        
        for i in range(n-count):
            v_r = self._rotate(v_r,self.raiz)

        v = np.copy(v_r)
        return self._divide(v,self.g)

        #for rotacionando a sindrome
        # se a sindrome In array
        # trocar a posicao 1 
        # invocar decode para novo ve
        pass



if __name__ == '__main__':
    import numpy as np

    G = np.genfromtxt('lab2_values/g10_6.csv', delimiter=',', dtype=int)
    H = np.genfromtxt('lab2_values/h10_6.csv', delimiter=',', dtype=int)
    ce = CyclicEncoder(G,H)
    u = np.array([1,0,1,0,0,0])
    print(u)
    v = ce.encode(u)
    e = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    ve = (v + e) % 2
    print(v)
    uf = ce.decode(ve)
    print(uf)
    # u = np.array([1,0,0,0,0,1])
    # h = np.array([1,1])
    # div = ce._divide(u,h)
    # print(div)
    # n = 10, 12, 14, 15, 16, 18, 20