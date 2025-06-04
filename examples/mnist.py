import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from extra.datasets import fetch_mnist

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

model = nn.Sequential(
  nn.Conv2d(1, 32, 5), nn.ReLU(),
  nn.Conv2d(32, 32, 5), nn.ReLU(),
  nn.BatchNorm2d(32), nn.MaxPool2d(2),
  nn.Conv2d(32, 64, 3), nn.ReLU(),
  nn.Conv2d(64, 64, 3), nn.ReLU(),
  nn.BatchNorm2d(64), nn.MaxPool2d(2),
  nn.Flatten(), nn.Linear(576, 10)
)
model = torch.jit.script(model).to(device)

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = fetch_mnist(tensor=True)
  X_train = X_train.float().div(255).view(-1, 1, 28, 28).to(device)
  X_test  = X_test.float().div(255).view(-1, 1, 28, 28).to(device)
  Y_train, Y_test = Y_train.to(device), Y_test.to(device)
  opt = torch.optim.Adam(model.parameters())
  
  def train_step():
    idx = torch.randint(0, X_train.shape[0], (int(os.getenv("BS", 512)),), device=device)
    loss = F.cross_entropy(model(X_train[idx]), Y_train[idx])
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss

  def get_test_acc():
    model.eval()
    with torch.no_grad():
      return (model(X_test).argmax(dim=1) == Y_test).float().mean().item()

  from tqdm import trange
  acc = float('nan')
  for i in (t:=trange(int(os.getenv("STEPS", 100)))):
    model.train()
    loss = train_step()
    if i % 10 == 9: acc = get_test_acc()
    t.set_description(f"loss: {loss.item():.2f} test_accuracy: {acc*100:.2f}%")