from numpy import *

class Adagrad(): 

    def __init__(self, dim, r=None):
        self.dim = dim
        self.eps = 1e-3

        # initial learning rate
        #self.learning_rate = 0.05  #original 100-dim depnn
        if r == None:
            self.learning_rate = 0.0001  #crf 100-dim
        else: 
            self.learning_rate = r
        #self.learning_rate = 0.02
        # stores sum of squared gradients 
        self.h = zeros(self.dim)

    def rescale_update(self, gradient):
        curr_rate = zeros(self.h.shape)
        self.h += gradient ** 2
        curr_rate = self.learning_rate / (sqrt(self.h) + self.eps)
        return curr_rate * gradient

    def reset_weights(self):
        self.h = zeros(self.dim)
