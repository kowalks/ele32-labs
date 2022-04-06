import random
import numpy as np

class Channel:
    name = 'Abstract channel'

    def transmit(self, word):
        raise NotImplementedError()


class BSC(Channel):
    name = 'BSC channel'

    def __init__(self, p=0.1):
        self.p = p
        self.name = f'BSC with p={p}'
    
    def transmit(self, word):
        k = len(word)
        p = self.p
        noise = random.choices([1,0], cum_weights=[p,1], k=k)
        bits = (noise + word) % 2
        return bits

class IdealChannel(Channel):
    name = 'Ideal channel'

    def transmit(self, word):
        return word

if __name__ == '__main__':
    word = np.array([0,1,1,1,0,1])
    channel = BSC()
    transmitted = channel.transmit(word)
    print(transmitted)