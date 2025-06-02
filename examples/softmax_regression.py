import numpy as np
np.set_printoptions(suppress=True)
from tqdm import trange
from extra.datasets import fetch_mnist

def principal_components(X, n_pcs=2):
  feature_means = X.mean(axis=0)
  X_centered = X - feature_means
  _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
  components = Vt.T[:, :n_pcs]
  return components, feature_means

class BCELoss:
  def forward(self, y, yhat):
    return -(y * np.log(yhat + 1e-9) + (1 - y) * np.log(1 - yhat + 1e-9)).mean()
  
  __call__ = forward

  def backward(self, x, y, yhat):
    diff = yhat - y
    grad_theta = x.T @ diff / len(x) 
    grad_theta_0 = diff.mean() 
    return grad_theta, grad_theta_0

class LogisticRegression:
  def __init__(self, in_features, lr=0.1):
    self.lr = lr
    self.theta, self.theta_0 = np.zeros(in_features, np.float32), 0.0

  def forward(self, x):
    return self.sigmoid(x @ self.theta + self.theta_0)
  
  __call__ = forward

  def sigmoid(self, z):
    return 1.0 / (1.0 + np.exp(-z))

  def step(self, grad_theta, grad_theta_0):
    self.theta -= self.lr * grad_theta 
    self.theta_0  -= self.lr * grad_theta_0

class NLLLoss:  
  def forward(self, y, yhat):
    return -np.mean(np.log(np.clip(yhat[np.arange(y.shape[0]), y], 1e-12, 1.0)))

  __call__ = forward

  def backward(self, x, y, yhat):
    n, _ = yhat.shape
    Y_onehot = np.zeros_like(yhat)
    Y_onehot[np.arange(n), y] = 1
    dZ = (yhat - Y_onehot) / n 
    grad_theta = dZ.T @ x
    grad_theta_0 = np.sum(dZ, axis=0)
    return grad_theta, grad_theta_0
    
class SoftmaxRegression:
  def __init__(self, in_features, out_features, lr=0.1, temperature=1, lambda_=0.1):
    self.lr = lr
    self.temperature = temperature
    self.lambda_ = lambda_
    self.theta = np.zeros((out_features, in_features), np.float32)
    self.theta_0 = np.zeros(out_features)
  
  def forward(self, x):
    return self.softmax(x @ self.theta.T + self.theta_0)
  
  __call__ = forward

  def softmax(self, z):
    z = z / self.temperature
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

  def step(self, grad_theta, grad_theta_0):
    if self.lambda_ > 0: grad_theta += self.lambda_ * self.theta # L2 Regularization
    self.theta -= self.lr * grad_theta 
    self.theta_0  -= self.lr * grad_theta_0

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = fetch_mnist()
  X_train, X_test = X_train / 255.0, X_test / 255.0
  
  pcs, feature_means = principal_components(X_train, n_pcs=75)
  X_train = (X_train - feature_means).dot(pcs)
  X_test = (X_test - feature_means).dot(pcs)

  """
  Y_train_8, Y_test_8 = (Y_train == 8).astype(int), (Y_test == 8).astype(int) # binary classifier
  model = LogisticRegression(in_features=784, lr=0.1)
  loss_fn = BCELoss()
  epochs = 5000
  for epoch in (t:=trange(epochs)):   
    samp = np.random.choice(X_train.shape[0], size=512)                         
    X, y = X_train[samp], Y_train_8[samp]
    out = model(X)
    loss = loss_fn(y, out)
    model.step(*loss_fn.backward(X, y, out))
    if epoch % 10 == 0 or epoch+1 == epochs:                              
      t.set_description(f'Epoch [{epoch}/{epochs}] loss: {loss:.4f} test_accuracy: {((model(X_test)>0.5).astype(int)==Y_test_8).mean():.4f}')
  """

  # multinomial model, extends logistic regression to handle multiple classes
  model = SoftmaxRegression(in_features=X_train.shape[1], out_features=10, lr=0.5, temperature=0.1, lambda_=0.0)
  loss_fn = NLLLoss()
  epochs = 10000
  for epoch in (t:=trange(epochs)):   
    samp = np.random.choice(X_train.shape[0], size=1024)                         
    X, y = X_train[samp], Y_train[samp]
    out = model(X)
    loss = loss_fn(y, out)
    model.step(*loss_fn.backward(X, y, out))
    if epoch % 50 == 0 or epoch+1 == epochs:                              
      t.set_description(f'Epoch [{epoch}/{epochs}] loss: {loss:.4f} test_accuracy: {(np.argmax(model(X_test), axis=1) == Y_test).mean()}')
  