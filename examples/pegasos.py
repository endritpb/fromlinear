# https://home.ttic.edu/~nati/Publications/PegasosMPB.pdf
import numpy as np
from tqdm import tqdm
from extra.datasets import fetch_amazon_polarity

def hinge_loss(X, Y, theta, theta_0):
  return np.maximum(0, 1-Y*(X@theta+theta_0)).mean()

# NOTE: will strongly bias the selection towards particular kinds of classifiers
# We want to find a new weight vector theta_{t+1} that is:
#	•	Close to the old weights theta (don’t want big changes), and
#	•	Correctly classifies the current example with margin at least 1
def passive_aggressive(X, Y, T=10, C=1, variant="PA"):
  # linear classifier that explicitly minimizes the hinge loss
  # C = 1/lambda
  match variant:
    case "PA": fn = lambda l, norm_sq: l / norm_sq
    case "PA-I": fn = lambda l, norm_sq: min(C, l / norm_sq)
    case "PA-II": fn = lambda l, norm_sq: l / (norm_sq + (0.5 / C))
    case _: raise ValueError(f"{variant} not known!")
  theta, theta_0 = np.zeros(X.shape[1], np.float32), 0.0
  for t in range(T):
    for i in tqdm(np.random.permutation(X.shape[0]), desc=f"Epoch [{t+1}/{T}]"):
      norm_sq = np.dot(X[i],X[i])
      if norm_sq == 0: continue  
      eta = fn(hinge_loss(X[i],Y[i],theta,theta_0), norm_sq)
      theta += eta*Y[i]*X[i]; theta_0 += eta*Y[i]
  return theta, theta_0

# Learning a classifier as an optimization problem
def pegasos(X, Y, lambda_, T=100, batch_size=32):
  theta, theta_0 = np.zeros(X.shape[1]), 0.0
  for t in tqdm(range(1, T+1)):
    indices = np.random.choice(X.shape[0], size=batch_size, replace=False)
    X_b, Y_b = X[indices], Y[indices]
    A = np.where(Y_b*(X_b@theta+theta_0) < 1)[0]
    eta = 1.0 / (lambda_ * t)
    if len(A) > 0:
      grad_theta, grad_theta_0 = np.sum(Y_b[A, None]*X_b[A], axis=0), np.sum(Y_b[A])
    else:
      grad_theta, grad_theta_0 = 0.0, 0.0
    theta = (1-eta*lambda_)*theta + (eta/batch_size) * grad_theta
    theta_0 = theta_0 + (eta/batch_size) * grad_theta_0  
    w_norm = np.linalg.norm(theta)
    if w_norm != 0:
      theta = min(1.0, 1.0 / (np.sqrt(lambda_) * w_norm)) * theta
  return theta, theta_0

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = fetch_amazon_polarity()
  Y_train, Y_test = np.where(Y_train==0,-1,1), np.where(Y_test==0,-1,1)
  print(X_train.shape)

  variant = "PA-I"
  theta, theta_0 = passive_aggressive(X_train, Y_train, T=10, C=1, variant=variant)
  print(f"[Passive-Aggressive {variant}] test_accuracy: {(np.sign(X_test@theta+theta_0)==Y_test).mean():.4f}\n")

  theta, theta_0 = pegasos(X_train, Y_train, T=5000, lambda_=1e-5, batch_size=32)
  print(f"[Mini-Batch Pegasos] test_accuracy: {(np.sign(X_test@theta+theta_0)==Y_test).mean():.4f}\n")
  
   