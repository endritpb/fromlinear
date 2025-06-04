import os
import torch
import torch.nn as nn
from extra.datasets import fetch_mnist

model = nn.Sequential(nn.Linear(784, 546), nn.LeakyReLU(), nn.Linear(546, 10))

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = fetch_mnist(tensor=True)
  X_train, X_test = X_train/255.0, X_test/255.0
  opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
  loss_fn = nn.CrossEntropyLoss()
  def train_step():
    model.train()
    opt.zero_grad()
    idx = torch.randint(0, X_train.shape[0], (int(os.getenv("BS", 256)),))
    loss = loss_fn(model(X_train[idx]), Y_train[idx])
    loss.backward()
    opt.step()
    return loss

  def get_test_acc():
    model.eval()
    with torch.no_grad():
      return (model(X_test).argmax(dim=1) == Y_test).float().mean().item()

  from tqdm import trange
  acc = float('nan')
  for i in (t:=trange(int(os.getenv("STEPS", 100)))):
    loss = train_step()
    if i % 10 == 9: acc = get_test_acc()
    t.set_description(f"loss: {loss.item():.2f} test_accuracy: {acc*100:.2f}%")