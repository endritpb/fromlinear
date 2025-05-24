# https://cs.nyu.edu/~mohri/pub/pmb.pdf
import os, numpy as np
from tqdm import tqdm
from extra.datasets import fetch_amazon_polarity

def perceptron(X, Y, T=10):
  theta, theta_0 = np.zeros(X.shape[1], np.float32), 0.0
  for t in range(T):
    k = 0
    for i in tqdm(np.random.permutation(X.shape[0]), desc=f"Epoch [{t+1}/{T}]"):
      if Y[i]*(X[i]@theta+theta_0) <= 1e-8:
        theta += Y[i]*X[i]; theta_0 += Y[i]
        k += 1
  return theta, theta_0, k

# We can tweak the algorithm to extract a decent classifier using averaged perceptron
def averaged_perceptron(X, Y, T=10):
  theta, theta_0 = np.zeros(X.shape[1], np.float32), 0.0       
  theta_bar, theta_0_bar, C = np.zeros_like(theta), 0.0, 1       
  stp = 1.0 / (X.shape[0] * T)
  for t in range(T):                             
    for i in tqdm(np.random.permutation(X.shape[0]), desc=f"Epoch [{t+1}/{T}]"):
      if Y[i]*(theta@X[i]+theta_0) <= 0:   
        theta += Y[i]*X[i]; theta_0 += Y[i]                          
        theta_bar += C*Y[i]*X[i]; theta_0_bar += C*Y[i]                  
      C -= stp                      
  return theta_bar, theta_0_bar                  

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = fetch_amazon_polarity()
  Y_train, Y_test = np.where(Y_train==0,-1,1), np.where(Y_test==0,-1,1)
  print(X_train.shape)
  theta, theta_0, mistakes = perceptron(X_train, Y_train, T=5)
  print(f"[Realizable Case Perceptron] test_accuracy: {(np.sign(X_test@theta+theta_0)==Y_test).mean():.4f}")
  if os.getenv("VERBOSE")!=None:
    print("\n[Perceptron] Convergence-theorem:")
    print(f"  L1-NORM ‖θ‖₁ : {np.sum(np.abs(theta)):12.4f}")
    print(f"  L2-NORM ‖θ‖₂ : {np.linalg.norm(theta, ord=2):12.4f}")
    R = np.linalg.norm(X_train, axis=1).max()
    gamma = 1/np.linalg.norm(theta, ord=2)
    print(f"  R={R:2f}; gamma={gamma:2f}")
    print(f"  mistakes<=(R/gamma)^2 {mistakes}<={int((R/gamma)**2)} -> {mistakes<=(R/gamma)**2} \n")
  theta_bar, theta_bar_0 = averaged_perceptron(X_train, Y_train, T=5)
  print(f"[Averaged Perceptron] test_accuracy: {(np.sign(X_test@theta_bar+theta_bar_0)==Y_test).mean():.4f}")