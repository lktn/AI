import numpy as np

class Model:
    def __init__(self, activation):
        n = 10
        self.activation = activation
        self.W1 = np.random.randn(2, n)
        self.B1 = np.zeros((1, n))
        self.W2 = np.random.randn(n, n)
        self.B2 = np.zeros((1, n))
        self.W3 = np.random.randn(n, 1)
        self.B3 = np.zeros((1, 1))

    def func(self, Z):
        if self.activation == 'Sigmoid':
            return 1/(1+np.exp(-Z))
        if self.activation == 'tanh':
            return np.tanh(Z)
        if self.activation == 'ReLU':
            return np.maximum(0, Z)

    def d_func(self, A):
        if self.activation == 'Sigmoid':
            return A*(1-A)
        if self.activation == 'tanh':
            return 1-A**2
        if self.activation == 'ReLU':
            return 1.0 * (A > 0)

    def forward(self, x):
        self.Z1 = x @ self.W1 + self.B1
        self.A1 = self.func(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.B2
        self.A2 = self.func(self.Z2)
        self.Z3 = self.A2 @ self.W3 + self.B3
        return self.func(self.Z3)

    def train(self, x_train, y_train, epoch, learning_rate):
        for i in range(epoch+1):
            tong_loss = 0
            for x, y in zip(x_train, y_train):
                x = x.reshape(1, -1)
                y = y.reshape(1, -1)

                self.A3 = self.forward(x)

                loss = np.mean((self.A3 - y) ** 2)
                tong_loss += loss

                d_A3 = self.A3 - y

                d_Z3 = d_A3 * self.d_func(self.A3)
                d_W3 = self.A2.T @ d_Z3
                d_B3 = d_Z3

                d_A2 = d_Z3 @ self.W3.T
                
                d_Z2 = d_A2 * self.d_func(self.A2)
                d_W2 = self.A1.T @ d_Z2
                d_B2 = d_Z2

                d_A1 = d_Z2 @ self.W2.T

                d_Z1 = d_A1 * self.d_func(self.A1)
                d_W1 = x.T @ d_Z1 
                d_B1 = d_Z1

                self.W3 -= learning_rate * d_W3
                self.B3 -= learning_rate * d_B3
                self.W2 -= learning_rate * d_W2
                self.B2 -= learning_rate * d_B2
                self.W1 -= learning_rate * d_W1
                self.B1 -= learning_rate * d_B1

            if i % 200 == 0:
                print("Epoch:", i, "Loss:", tong_loss)

x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
y_train = np.array([[0],   [1],   [1],   [0]])

Mohinh = Model(activation='Sigmoid')
Mohinh.train(x_train=x_train, y_train=y_train, epoch=100000, learning_rate=0.1)
