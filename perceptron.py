#!/usr/bin/env python3
# https://arxiv.org/pdf/1305.0208
import numpy as np

rng = np.random.default_rng(42)

def load_data(n_per_class:int=100, realizable_case:bool=True):
  scale, pt = (0.95, 2) if realizable_case else (1.6, 1.1)
  pos = rng.normal(loc=0.0, scale=scale, size=(n_per_class, 3)) + np.array([[pt]*3])
  neg = rng.normal(loc=0.0, scale=scale, size=(n_per_class, 3)) + np.array([[-pt]*3])
  X = np.vstack([pos, neg]) 
  y = np.hstack([np.ones(n_per_class, dtype=int), -np.ones(n_per_class, dtype=int)])
  return X, y

def perceptron(X:np.ndarray, y:np.ndarray, T:int=10):
  n, d = X.shape
  theta, theta_0, k_updates = np.zeros(d, dtype=np.float16), 0., 0
  for _ in range(T):
    e = 0
    for i in np.random.permutation(n):
      if y[i] * np.sign(theta @ X[i] + theta_0) <= 1e-8:
        theta += y[i]*X[i]
        theta_0 += y[i]
        e += 1; k_updates += 1    
    if e == 0: break
  return theta, theta_0, k_updates
      
if __name__ == "__main__":
  X, y = load_data(n_per_class=500, realizable_case=True) 
  theta, theta_0, k_updates = perceptron(X=X, y=y, T=10)
  theta_norm = float(np.linalg.norm(theta))
  R = float(np.max(np.linalg.norm(X, axis=1)))
  gamma = float(np.min(y * (X @ theta + theta_0)) / theta_norm)
  bound = (R / gamma)**2 if gamma > 0 else float('inf')
  print(f"k-updates={k_updates}\nR = {R:.6f}\ngamma = {gamma:.6f}\n(R/gamma)^2 = {bound:.6f}")

  # decision boundary plot
  '''
  import matplotlib.pyplot as plt
  
  offset = gamma * theta_norm if gamma > 0 else 0
  xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 40), np.linspace(X[:, 1].min(), X[:, 1].max(), 40))
  zz_sep = (-(theta[0] * xx + theta[1] * yy + theta_0)) / theta[2]
  zz_plus = ((offset - theta[0] * xx - theta[1] * yy - theta_0) / theta[2])
  zz_minus = ((-offset - theta[0] * xx - theta[1] * yy - theta_0) / theta[2])
  u = np.linspace(0, 2 * np.pi, 60)
  v = np.linspace(0, np.pi, 30)
  xs = R * np.outer(np.cos(u), np.sin(v))
  ys = R * np.outer(np.sin(u), np.sin(v))
  zs = R * np.outer(np.ones_like(u), np.cos(v))

  fig = plt.figure(figsize=(7, 6))
  ax = fig.add_subplot(projection="3d")
  ax.set_box_aspect((1, 1, 1))
  ax.scatter(*X[y==1].T, s=14, label="+1")
  ax.scatter(*X[y==-1].T, s=14, label="-1")
  ax.plot_surface(xx, yy, zz_sep, alpha=0.4, linewidth=0, antialiased=True) 
  ax.plot_surface(xx, yy, zz_plus, alpha=0.15, linewidth=0, antialiased=True)
  ax.plot_surface(xx, yy, zz_minus, alpha=0.15, linewidth=0, antialiased=True)
  ax.plot_wireframe(xs, ys, zs, linewidth=0.5, alpha=0.35)
  ax.set(xlim=(-R, R), ylim=(-R, R), zlim=(-R, R), xlabel="x1", ylabel="x2", zlabel="x3",
    title=f"R = {R:.3f},  gamma = {gamma:.3f},  bound = (R/gamma)^2 = {bound:.2f}")
  ax.legend()
  ax.grid(True)
  plt.tight_layout()
  plt.show()
  '''
