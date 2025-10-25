#!/usr/bin/env python3
# https://arxiv.org/pdf/1305.0208
import numpy as np

rng = np.random.default_rng(42)

def load_data(n_per_class:int=100):
  pos = rng.normal(loc=0.0, scale=0.8, size=(n_per_class, 3)) + np.array([[2.5]*3])
  neg = rng.normal(loc=0.0, scale=0.8, size=(n_per_class, 3)) + np.array([[-2.5]*3])
  X = np.vstack([pos, neg]) 
  y = np.hstack([np.ones(n_per_class, dtype=int), -np.ones(n_per_class, dtype=int)])
  return X, y

def perceptron(X:np.ndarray, y:np.ndarray, T:int=10):
  n, d = X.shape
  theta, theta_0 = np.zeros(d, dtype=np.float16), 0.
  for _ in range(T):
    for i in np.random.permutation(n):
      if y[i]*np.sign(theta.dot(X[i]) + theta_0) <= 1e-8:
        theta += y[i]*X[i]
        theta_0 += y[i]
  return theta, theta_0
      
if __name__ == "__main__":
  X, y = load_data(n_per_class=100) 
  theta, theta_0 = perceptron(X=X, y=y, T=5)
  
  # decision boundary plot
  '''
  import matplotlib.pyplot as plt
  ax = plt.figure().add_subplot(projection='3d')
  ax.scatter(*X[y==1].T, c="b", label="1", s=18)
  ax.scatter(*X[y==-1].T, c="r", label="-1", s=18)
  xx, yy = np.meshgrid(np.linspace(X[:,0].min(), X[:,0].max(), 10), np.linspace(X[:,1].min(), X[:,1].max(), 10))
  ax.plot_surface(xx, yy, -(theta[0]*xx + theta[1]*yy + theta_0)/theta[2], alpha=0.3)
  plt.show()
  '''
