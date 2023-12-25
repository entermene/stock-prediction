import numpy as np
import pandas as pd
# import feature_and_preprocessing3
# %run feature_and_preprocessing3.py
# import feature_and_preprocessing2
# %run feature_and_preprocessing2.py
from feature_and_preprocessing2 import y
# from feature_and_preprocessing3 import y1
from feature_and_preprocessing2 import expanded_inputs2
# from feature_and_preprocessing import expanded_inputs2
# from feature_and_preprocessing import expanded_inputs3

from feature_and_preprocessing2 import synaptic_weights

from feature_and_preprocessing3 import test_expanded_inputs2




class NeuralNetwork:
    def __init__(self, expanded_inputs2, y):
        self.input = expanded_inputs2
        self.weights1 = synaptic_weights
        self.y = y
        self.OutY = y

    def feedforward(self):
        inputY = self.y
        inputY2 = self.y

        I = np.identity(len(self.input.T))

        square = []
        
        square1 = []

        lamb = 0.1

        for i in range(1):
            # Transpose of the inverse matrix ((M^tM)^-1 M^t)^t y
            d_weights2 = (
                np.linalg.inv(self.input.T.dot(self.input) + lamb * I)
                .dot(self.input.T)
                .dot(inputY2)
            )
            d_weights1 = (
                np.linalg.inv(self.input.T.dot(self.input) + lamb * I)
                .dot(self.input.T)
                .dot(inputY)
            )

            self.OutY = self.input.dot(d_weights1)
            self.OutY2 = self.input.dot(d_weights2)
            


                                  
            square.append(np.sum((inputY2- self.OutY2) ** 2))
            square1.append(np.sum((inputY - y) ** 2))

            # print(f'Weights = {d_weights1}')
            # print(f'Square = {square}')
            print(f'Square = {square}')
            print(f'Square1 = {square1}')

            inputY2 = self.OutY2
        return d_weights1, d_weights2


if __name__ == "__main__":
    nn = NeuralNetwork(expanded_inputs2, y)

#     weights = nn.feedforward()
    
#     outputs=expanded_inputs2.dot(weights)
    
    weights,weights2 = nn.feedforward()
    
    outputs=expanded_inputs2.dot(weights)
    
    outputs2=expanded_inputs2.dot(weights2)
        
    outputs4=test_expanded_inputs2.dot(weights2)