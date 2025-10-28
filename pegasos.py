#!/usr/bin/env python3
# https://home.ttic.edu/~nati/Publications/PegasosMPB.pdf
import numpy as np
from extra.datasets import fetch_mnist

rng = np.random.default_rng(42)

def lossh(X:np.ndarray, y:np.ndarray, theta:np.ndarray, theta_0:float):
  return np.maximum(0.0, 1.0-y*(X@theta + theta_0))

# minimizes hinge loss while keeping parameters close to prior values
def passive_aggressive(X:np.ndarray, y:np.ndarray, T:int=10, variant:str="PA", C:float=1.0):
  assert variant in ("PA", "PA-I", "PA-II"), f"wrong variant name {variant}"
  n, d = X.shape
  theta, theta_0 = np.zeros(d, dtype=np.float32), 0.0
  tau_fn = {"PA": lambda l,n: l/n, "PA-I": lambda l,n: min(C, l/n), "PA-II": lambda l,n: l/(n+1/(2*C))}
  for _ in range(T):
    for i in rng.permutation(n):
      xi, yi = X[i], y[i]
      tau = tau_fn[variant](lossh(xi, yi, theta, theta_0).item(), np.linalg.norm(xi)**2+1e-12)
      theta += tau*yi*xi
      theta_0 += tau*yi
  return theta, theta_0

# mini-batch pegasos
def pegasos_bin(X:np.ndarray, y:np.ndarray, T:int=100_000, batch_size:int=512, _lambda:float=3e-1, project:bool=True):
  n, d = X.shape
  theta, theta_0 = np.zeros(d, np.float32), 0.0
  for t in range(1, T+1):
    indices = np.random.choice(n, size=min(batch_size, n), replace=False)
    Xb, yb = X[indices], y[indices]
    eta = 1.0/(_lambda*t)
    m = yb*(Xb@theta + theta_0) < 1.0
    theta = (1-eta*_lambda)*theta
    if np.any(m):
      theta += eta*(yb[m,None]*Xb[m]).mean(0)
      theta_0 += eta*yb[m].mean()
    if project:
      nrm = np.linalg.norm(theta); cap = 1/np.sqrt(_lambda)
      if nrm > cap: theta *= cap/nrm
  return theta, theta_0

class PegasosOvR:
  def __init__(self, _lambda:float=3e-1, batch_size:int=512, project:bool=False):
    self._lambda = _lambda
    self.batch_size = batch_size
    self.project = project
    self.classes_ = None
    self.W_, self.b_ = None, None

  def train(self, X:np.ndarray, y:np.ndarray, T:int=1000):
    self.classes_ = np.unique(y)
    K, d = len(self.classes_), X.shape[1]
    W, b = np.zeros((K, d), np.float32), np.zeros(K, np.float32)
    for k, c in enumerate(self.classes_):
      W[k], b[k] = pegasos_bin(X=X, y=np.where(y==c,1,-1), T=T, batch_size=self.batch_size, _lambda=self._lambda, project=self.project)
    self.W_, self.b_ = W, b
    return self
  
if __name__ == "__main__":
  X_train, y_train, X_test, y_test = fetch_mnist()
  mu = X_train.mean(0, keepdims=True); sd = X_train.std(0, keepdims=True) + 1e-8
  X_train /= np.linalg.norm(((X_train.astype(np.float32) - mu)/sd), axis=1, keepdims=True) + 1e-8
  X_test /= np.linalg.norm(((X_test.astype(np.float32) - mu)/sd), axis=1, keepdims=True) + 1e-8

  # variant = "PA-I"
  # theta, theta_0 = passive_aggressive(X_train, np.where(y_train==8,1,-1), T=10, variant=variant, C=0.9)
  # print(f"{variant}: accuracy={(np.where(y_train==8,1,-1) == np.where(X_test@theta + theta_0 >= 0.0, 1, -1)).mean():.4f}, " +
  #       f"lossh={lossh(X_test, np.where(y_train==8,1,-1), theta, theta_0).mean():.4f}")
  
  # theta, theta_0 = pegasos_bin(X_train, np.where(y_train==8,1,-1), T=2_000, batch_size=32, _lambda=3e-1, project=True)
  # print(f"Pegasos: accuracy={(np.where(y_test==8,1,-1) == np.where(X_test@theta + theta_0 >= 0.0, 1, -1)).mean():.4f}, " +
  #       f"lossh={lossh(X_test, np.where(y_test==8,1,-1), theta, theta_0).mean():.4f}")

  model = PegasosOvR(_lambda=3e-1, batch_size=256, project=True).train(X_train, y_train, T=50000)
  print(f"PegasosOvR: accuracy={(model.classes_[np.argmax(X_test@model.W_.T + model.b_, axis=1)]==y_test).mean():.4f}, " +
        f"lossh={np.mean([lossh(X_test, np.where(y_test == c, 1, -1), theta, theta_0) for c, (theta, theta_0) in zip(model.classes_, zip(model.W_, model.b_))]):.4f}")
