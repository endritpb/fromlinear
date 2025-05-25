# https://home.ttic.edu/~nati/Publications/PegasosMPB.pdf
import numpy as np
from tqdm import tqdm
from extra.helpers import sparse
from extra.datasets import fetch_amazon_polarity

# NOTE: will strongly bias the selection towards particular kinds of classifiers
# We want to find a new weight vector theta_{t+1} that is:
#	•	Close to the old weights theta (don’t want big changes), and
#	•	Correctly classifies the current example with margin at least 1
def passive_aggressive(X, Y, T=10, C=1, variant="PA"):
  # https://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf
  match variant:
    case "PA": fn = lambda l, norm_sq: l / norm_sq
    case "PA-I": fn = lambda l, norm_sq: min(C, l / norm_sq)
    case "PA-II": fn = lambda l, norm_sq: l / (norm_sq + (0.5 / C))
    case _: raise ValueError(f"{variant} not known!")
  theta, theta_0 = np.zeros(X.shape[1], np.float32), 0.0
  for t in range(T):
    for i in tqdm(np.random.permutation(X.shape[0]), desc=f"Epoch [{t+1}/{T}]"):
      xi = sparse(X[i])
      norm_sq = np.dot(xi, xi)
      if norm_sq == 0: continue  
      eta = fn(np.maximum(0, 1 - Y[i] * (xi @ theta + theta_0)).mean(), norm_sq) # lagrange multipler eta > 0
      theta += eta*Y[i]*xi
      theta_0 += eta*Y[i]
  return theta, theta_0

# Learning a classifier as an optimization problem
def pegasos(X, Y, T=100, lambda_=1e-4, batch_size=32):
  theta, theta_0 = np.zeros(X.shape[1]), 0.0
  for t in (pbar:=tqdm(range(1, T+1))):
    pbar.set_description(f"Epoch [{t-1}/{T}]")
    indices = np.random.choice(X.shape[0], size=batch_size, replace=False)
    X_b, Y_b = X[indices], Y[indices]
    A = np.where(Y_b * (X_b @ theta + theta_0) < 1)[0]
    eta = 1.0 / (lambda_ * t)
    if len(A) > 0:
      grad_theta = X_b[A].multiply(Y_b[A][:, None]).sum(axis=0).A1 if hasattr(X_b[A], "multiply") else np.sum(Y_b[A, None] * X_b[A], axis=0)
      grad_theta_0 = np.sum(Y_b[A])
    else:
      grad_theta = np.zeros_like(theta)
      grad_theta_0 = 0.0
    theta = (1 - eta * lambda_) * theta + (eta / batch_size) * grad_theta
    theta_0 = theta_0 + (eta / batch_size) * grad_theta_0
    w_norm = np.linalg.norm(theta)
    if w_norm != 0:
      theta = min(1.0, 1.0 / (np.sqrt(lambda_) * w_norm)) * theta
  return theta, theta_0

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = fetch_amazon_polarity()
  Y_train, Y_test = np.where(Y_train == 0, -1, 1), np.where(Y_test == 0, -1, 1)
  print(X_train.shape)

  variant = "PA-II"
  theta, theta_0 = passive_aggressive(X_train, Y_train, T=5, C=0.1, variant=variant)
  print(f"[Passive-Aggressive {variant}] test_accuracy: {(np.sign(X_test @ theta + theta_0) == Y_test).mean():.4f}\n")
  
  batch_size = 256
  theta, theta_0 = pegasos(X_train, Y_train, T=10*(X_train.shape[0]//batch_size), lambda_=5e-3, batch_size=batch_size)
  print(f"[Mini-Batch Pegasos] test_accuracy: {(np.sign(X_test @ theta + theta_0) == Y_test).mean():.4f}\n")
  