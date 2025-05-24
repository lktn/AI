import numpy as np

def ReLU(x):        return np.maximum(0, x)
def Sigmoid(x):     return 1/(1+np.exp(-x))
def LeakyReLU(x,a): return np.where(x > 0, x, a*x)
def Elu(x,a):       return np.where(x > 0, x, a*(np.exp(x)-1))

w = [] #weight
b = [] #bias

def Linear(x, y):
    w.append([np.random.uniform(-1, 1, x) for _ in range(y)])
    b.append(np.random.uniform(-1, 1, y))

def hidden(z):
    for i in range(len(z) - 1):
        Linear(z[i], z[i+1])

hidden([1, 5, 5, 1])

def f(x):
    x = np.array([x])
    for h in range(len(w)): #2
        x = np.array([x @ i for i in w[h]]) + b[h]
        if h == (len(w)-1): x = x
        else: x = LeakyReLU(x, 0.1)
    return x[0]

x = float(input("x = "))
out = f(x)
loss = (out-np.sin(x))**2/2
print("out = ", out, "loss = ", loss)