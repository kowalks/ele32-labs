import random
import numpy as np

from channel import BSC
from encoder import HammingEncoder

class Simulator:
    def __init__(self, encoder, channel, k, l):
        self.encoder = encoder
        self.channel = channel
        self.k = k
        self.l = l

    def generate_bits(self):
        words = []
        for _ in range(self.l):
            bits = random.choices([0,1], k=self.k)
            word = np.array(bits)
            words.append(word)
        return words

    def simulate(self):
        words = self.generate_bits()
        encoded = map(self.encoder.encode, words)
        transmitted = map(self.channel.transmit, encoded)
        decoded = map(self.encoder.decode, transmitted)

        # s = sum((x == y).all() for x, y in zip(words, decoded))
        s = sum((x == y).sum() for x, y in zip(words, decoded))
        n_bits = self.l * self.k
        return (n_bits - s)/n_bits


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    encoder = HammingEncoder()
    p = [5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]
    k = 4
    l = 1000000//k
    channels = map(BSC, p)
    sim = map(lambda channel: Simulator(encoder, channel, k=k, l=l), channels)
    s = map(Simulator.simulate, sim)

    s = list(s) # [0.93744, 0.424728, 0.149772, 0.045068, 0.007736, 0.00192, 0.00054, 8e-05, 2.8e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    print(s)
    
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(p, s)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()