import numpy as np
from sklearn.metrics import mean_squared_error
import pickle

class NeuralNetwork:
    def __init__(self, layer, learning_rate=0.01):
        self._layer = layer
        self._learning_rate = learning_rate
        self._params = {}
        self._values = {}
        self.initialize_weights()

    def relu(self, z):
        return np.maximum(0,z)
    
    def derivative_relu(self, z):
        return (z > 0).astype(int)

    def initialize_weights(self):
        for i in range(1, len(self._layer)):
            self._params[f'W{i}'] = np.random.randn(self._layer[i], self._layer[i-1])*0.01
            self._params[f'b{i}'] = np.random.randn(self._layer[i],1)*0.01

    def forward_propagation(self, X):
        for i in range(1, len(self._layer)):
            if i == 1:
                self._values[f'Z{i}'] = np.dot(self._params[f'W{i}'], X.T) + self._params[f'b{i}']
                self._values[f'A{i}'] = self.relu(self._values[f'Z{i}'])
            else:
                self._values[f'Z{i}'] = np.dot(self._params[f'W{i}'], 
                                               self._values[f'A{i-1}']) + self._params[f'b{i}']
                if i == len(self._layer) - 1:
                    self._values[f'A{i}'] = self._values[f'Z{i}']
                else:
                    self._values[f'A{i}'] = self.relu(self._values[f'Z{i}'])
        return self._values[f'A{len(self._layer)-1}']

    def compute_cost(self, y_pred):
        # RMSE
        rmse = mean_squared_error(self._y, y_pred.reshape(-1,1), squared=False)
        return rmse
    
    def derivative_loss(self, y_true, y_pred):
        return 2*(y_pred - y_true)/len(y_true)

    def backward_propagation(self):
        n = len(self._y.T)
        for i in range(len(self._layer)-1,0,-1):
            if i==len(self._layer)-1:
                dA = self.derivative_loss(self._y, self._values[f'A{i}'])
                dZ = dA
            else:
                dA = np.dot(self._params[f'W{i+1}'].T, dZ)
                dZ = np.multiply(dA, self.derivative_relu(self._values[f'A{i}']))
            if i==1:
                self._params[f'dW{i}'] = 1/n * np.dot(dZ, self._input)
            else:
                self._params[f'dW{i}'] = 1/n * np.dot(dZ,self._values[f'A{i-1}'].T)
            self._params[f'db{i}'] = 1/n * np.sum(dZ, axis=1, keepdims=True)
        for i in range(1, len(self._layer)):
            self._params[f'W{i}'] -= self._learning_rate * self._params[f'dW{i}']
            self._params[f'b{i}'] -= self._learning_rate * self._params[f'db{i}']

    def fit(self, X_train, y_train, iterations=10000):
        self._input = X_train
        self._y = y_train
        for i in range(iterations):
            y_pred = self.forward_propagation(X_train)
            rmse = self.compute_cost(y_pred)
            self.backward_propagation()
            if i % 100 == 0:
                print(f'Iter: {i+1}. RMSE: {rmse}.')
        print(f'Final: {i+1}. RMSE: {rmse}.')
    
    def predict(self, X_test):
        return self.forward_propagation(X_test).reshape(-1,1).ravel()

    def save_params(self):
        pickle.dump(self._params, open('NerualNetwork_weights.pkl', 'wb'))
    
    def load_params(self):
        file = open('NerualNetwork_weights.pkl', 'rb')
        self._params = pickle.load(file)
        file.close()

'''
layer = [X_train_normalized.shape[1], 128, 64, 1]  
nn = NeuralNetwork(layer)
nn.fit(X_train_normalized, y_train)
nn.save_params()
'''