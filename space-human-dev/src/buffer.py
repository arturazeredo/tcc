import numpy as np


class BufferR:
    def __init__(self, max=20):
        self.buffx = []
        self.buffy = []
        self.max = max

    def _sub(self):
        self.buffy.pop(-1)
        self.buffx.pop(-1)
        return True

    def _add(self, pts):
        x, y = pts

        if self.max:
            self.buffy.append(y)
            self.buffx.append(x)
        return True

    def medianX(self):
        return np.median(self.buffx)

    def mediaY(self):
        return np.median(self.buffy)


    def stdX(self):
        return np.std(self.buffx)

    def stdY(self):
        return np.std(self.buffy)





