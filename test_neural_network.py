import numpy as np

def ReLU(x):             return np.maximum(0, x)
def Sigmoid(x):          return 1/(1+np.exp(-x))
def Softmax(Z):          return np.exp(Z)/np.exp(Z).sum(axis = 0)
def LeakyReLU(x, a=0.1): return np.where(x > 0, x, a*x)
def Elu(x, a=0.1):       return np.where(x > 0, x, a*(np.exp(x)-1))

w = [] # weight
b = [] # bias
a = [] # activation function

def Linear(x, y):
    w.append([np.random.uniform(-1, 1, x) for _ in range(y)])
    b.append(np.random.uniform(-1, 1, y))

def forward(x):
    x = np.array([x])
    for h in range(len(w)):
        x = np.array([x @ i for i in w[h]]) + b[h]
        try: x = a[h](x)
        except: pass
    return x

def Network(z):
    a.clear()
    b.clear()
    w.clear()
    for i in range(len(z) - 1):
        Linear(z[i], z[i+1])

def Activation(c):
    a.clear() 
    a.extend(c)

Network([1, 5, 5, 1])
Activation([LeakyReLU, LeakyReLU])
