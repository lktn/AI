import numpy as np
import matplotlib.pyplot as plt
# chế mạng nơ ron nhan tạo từ đầu
# 26/8/2025
def Dense(n, activation=None):
    def Sigmoid(z):                return 1/(1+np.exp(-z))
    def d_Sigmoid(a):              return a*(1-a)
    def tanh(z):                   return np.tanh(z)
    def d_tanh(a):                 return 1 - a**2
    def ReLU(z):                   return np.maximum(0, z)
    def d_ReLU(a):                 return np.where(a > 0, 1, 0)
    def LeakyReLU(z, alpha=0.1):   return np.maximum(alpha*z, z)
    def d_LeakyReLU(a, alpha=0.1): return np.where(a > 0, 1, alpha)
    def ELU(z, alpha=1):           return np.where(z > 0, z, alpha*(np.exp(z)-1))
    def d_ELU(a, alpha=1):         return np.where(a > 0, 1, a + alpha)
    def SoftPlus(z):               return np.log(1 + np.exp(z))
    def d_SoftPlus(a):             return 1 - np.exp(-a)
    def SoftSign(z):               return z/(1+abs(z))
    def d_SoftSign(a):             return (1-abs(a))**2
    def Linear(z):                 return z
    def d_Linear(a):               return 1

    if activation == 'leakyrelu':  return [n, LeakyReLU, d_LeakyReLU]
    elif activation == 'sigmoid':  return [n, Sigmoid  , d_Sigmoid  ]
    elif activation == 'tanh':     return [n, tanh     , d_tanh     ]
    elif activation == 'relu':     return [n, ReLU     , d_ReLU     ]
    elif activation == 'elu':      return [n, ELU      , d_ELU      ]
    elif activation == 'softplus': return [n, SoftPlus , d_SoftPlus ]
    elif activation == 'softsign': return [n, SoftSign , d_SoftSign ]
    elif activation is None:       return [n, Linear   , d_Linear   ]

    raise ValueError("Không có hàm kích hoạt " + activation)

class Model:
    def __init__(self, layer, scale=0.2):
        self.W, self.B = {}, {}
        self.activation = []
        self.d_activation = []
        for i in range(1, len(layer)):
            self.W[f"W{i}"] = np.random.randn(layer[i-1][0], layer[i][0]) * scale
            self.B[f"B{i}"] = np.zeros((1, layer[i][0]))
            self.activation.append(layer[i][1])
            self.d_activation.append(layer[i][2])
        
    def compile(self, optimizer=None, loss='MSE'):
        if loss == 'MSE':     
            def MSE(y, y_pred):   return np.mean((y - y_pred)**2)
            def d_MSE(y, y_pred): return 2 * (y_pred - y)     
            self.cost, self.d_cost = MSE, d_MSE
        elif loss == 'BinaryCrossEntropy':
            def BinaryCrossEntropy(y, y_pred, eps=1e-9): return - (y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
            def d_BinaryCrossEntropy(y, y_pred):         return y_pred - y
            self.cost, self.d_cost = BinaryCrossEntropy, d_BinaryCrossEntropy

    def predict(self, input):
        input = input.reshape(-1, 1)
        for i in range(1, len(self.B) + 1):
            input = self.activation[i-1](input @ self.W[f"W{i}"] + self.B[f"B{i}"])
        return input

    def forward(self, input):
        self.Z = {}
        self.A = {}
        self.A["A0"] = input
        for i in range(1, len(self.B) + 1):
            self.Z[f"Z{i}"] = self.A[f"A{i-1}"] @ self.W[f"W{i}"] + self.B[f"B{i}"]
            self.A[f"A{i}"] = self.activation[i-1](self.Z[f"Z{i}"])
        return self.A[f"A{len(self.B)}"]

    def train(self, epochs, x_train, y_train, lr=0.01):
        n_samples = len(x_train)
        for epoch in range(epochs):
            loss = 0
            for x, y in zip(x_train, y_train):
                x = x.reshape(1, -1)
                y = y.reshape(1, -1)
                y_pred = self.forward(x)
                loss += self.cost(y, y_pred)

                d = self.d_cost(y, y_pred)
                d = d * self.d_activation[-1](self.A[f"A{len(self.B)}"]) 
                dW = self.A[f"A{len(self.B)-1}"].T @ d
                dB = d
                self.W[f"W{len(self.B)}"] -= lr * dW
                self.B[f"B{len(self.B)}"] -= lr * dB

                for i in range(len(self.B) - 1, 0, -1):
                    d = (d @ self.W[f"W{i+1}"].T) * self.d_activation[i-1](self.A[f"A{i}"])
                    dW, dB = self.A[f"A{i-1}"].T @ d, d
                    self.W[f"W{i}"] -= lr * dW
                    self.B[f"B{i}"] -= lr * dB

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss = {loss/n_samples:.10f}")
                plt.clf()
                plt.plot(x_train, y_train)
                plt.axis('equal')
                plt.plot(x_train, my_model.predict(x_train))
                plt.legend()
                plt.pause(0.001)

x_train = np.linspace(-6, 6, 200)
y_train = np.sin(x_train)

my_model = Model([
    Dense(1),
    Dense(30, activation='tanh'),
    Dense(30, activation='tanh'),
    Dense(30, activation='tanh'),
    Dense(1, activation='tanh'),
])
my_model.compile(loss='MSE')

my_model.train(x_train=x_train, y_train=y_train, epochs=10000, lr=0.001)

